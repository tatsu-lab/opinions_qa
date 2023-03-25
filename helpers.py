import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PEW_SURVEY_LIST = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92] 
DEMOGRAPHIC_ATTRIBUTES = ['Overall',
 'CREGION',
 'AGE',
 'SEX',
 'EDUCATION',
 'CITIZEN',
 'MARITAL',
 'RELIG',
 'RELIGATTEND',
 'POLPARTY',
 'INCOME',
 'POLIDEOLOGY',
 'RACE']


MODEL_NAMES = {'human max': 'human (worst)',
               'human mean': 'human (avg)',
               'random': 'random',
               'ai21_j1-grande': 'j1-grande',
               'ai21_j1-jumbo': 'j1-jumbo',
               'ai21_j1-grande-v2-beta': 'j1-grande-v2-beta',
               'openai_ada': 'ada', 
               'openai_davinci': 'davinci', 
               'openai_text-ada-001': 'text-ada-001', 
               'openai_text-davinci-001': 'text-davinci-001', 
               'openai_text-davinci-002': 'text-davinci-002', 
               'openai_text-davinci-003': 'text-davinci-003',
               }

MODEL_ORDER = {k: ki for ki, k in enumerate(MODEL_NAMES.keys())}

def get_probabilities(lps, references, mapping):

    min_prob = np.exp(np.min(list(lps.values())))
    remaining_prob = max(0, 1 - sum([np.exp(v) for v in lps.values()]))
    
    dist, misses = [], []
    for ref in references:
        prefix = mapping[ref]
        values = [lps[key] for key in [f" {prefix}", prefix] if key in lps]
        misses.append(len(values) == 0)
        dist.append(np.max(values) if len(values) else None)

    Nmisses = sum(misses)
    if Nmisses > 0:
        miss_value = np.log(min(min_prob, remaining_prob / Nmisses))
        dist = [d if d is not None else miss_value for d in dist]
    
    probs_unnorm = np.array([np.exp(v) for v in dist])
    
    res = {'logprobs': dist,
           'probs_unnorm': probs_unnorm,
           'probs_norm': probs_unnorm / np.sum(probs_unnorm),
           'misses': misses}
           
    return res
    
def extract_model_opinions(result_instance, context_type, info_df):
        
    row = {}
    
    input_id = result_instance['instance']['id']    
    question_raw = result_instance['instance']['input']['text']
    references = [r['output']['text'] for r in result_instance['instance']['references']]
    mapping = result_instance['output_mapping']
    if context_type not in ['steer-portray', 'steer-bio']:
        context = result_instance['request']['prompt'].split(f"Question: {question_raw}")[0].strip()
    else:
        context = question_raw.split('Question:')[0].strip() + '\n'
        question_raw = question_raw.replace(context, "").strip().replace('Question: ', '')
    question = question_raw + f" [{'/'.join(references)}]"
    
    top_k_logprobs = result_instance['result']['completions'][0]['tokens'][0]['top_logprobs']

    for k, v in zip(['input_id', 'question_raw', 'question', 'references', 
                     'context', 'mapping', 'top_k_logprobs'],
                     [input_id, question_raw, question, references, context, mapping, top_k_logprobs]):
        row[k] = v
        
    ## Get probability distribution
    
    info_loc = np.where(np.logical_and(info_df['question'] == question_raw,
                                       [set(r) == set(references) for r in info_df['references']]))[0]
    assert len(info_loc) == 1

    info = info_df.iloc[info_loc]
    ordinal = info['option_ordinal'].values[0]
    ordinal_refs = info['references'].values[0][:len(ordinal)]
    refusal_refs = info['references'].values[0][len(ordinal):]
    
    dist_info = get_probabilities(top_k_logprobs, info['references'].values[0], {v: k for k, v in mapping.items()})
    dist_info['D_M'] = dist_info['probs_unnorm'][:len(ordinal)] / np.sum(dist_info['probs_unnorm'][:len(ordinal)])
    dist_info['R_M'] = np.sum(dist_info['probs_norm'][len(ordinal):])
    dist_info['ordinal'] = ordinal
    dist_info['ordinal_refs'] = ordinal_refs
    dist_info['refusal_refs'] = refusal_refs
    dist_info['qkey'] = info['key'].values[0]
        
    row.update(dist_info)
        
    return row

def extract_human_opinions(hdf, model_df, md_df, demographic='Overall', wave=None):
    
    assert wave is not None
        
    question_keys = list(set(model_df['qkey']))
    weight_key = [w for w in hdf.columns if w == f'WEIGHT_W{wave}']
    assert len(weight_key) == 1
    weight_key = weight_key[0]
    
    
    res = {'qkey': [], 'attribute': [], 'group': [], 'D_H': [], 'R_H': []}
    
    for qkey in question_keys:
        col_names = [qkey, demographic] if demographic != 'Overall' else [qkey]
        
        cdf = hdf[[weight_key] + col_names]
        cdf = cdf[[type(v) == str for v in cdf[qkey]]]
        cdf = cdf.groupby(col_names, as_index=False).agg({weight_key: sum})
        
        if demographic == 'Overall':
            dist_all = {'Overall': {k: v for k, v in zip(cdf[qkey], cdf[weight_key])}}
        else:
            options = md_df[md_df['key'] == demographic]['options'].values[0]
            
            def chain(row):
                dist = {k: v for k, v in zip(row[qkey], row[weight_key])}
                row['dist'] = dist
                return row
            cdf = cdf[cdf[demographic].isin(options)]
            cdf = cdf.groupby([demographic], as_index=False).agg(list).apply(chain, axis=1)
            dist_all = {k: v for k, v in zip(cdf[demographic], cdf['dist'])}
        
        vdf = model_df[model_df['qkey'] == qkey][['ordinal_refs', 'refusal_refs', 'ordinal']].iloc[:1]
        
        for group_name, dist in dist_all.items():
            opinion_dist = np.array([dist[v] if v in dist else 0 for v in vdf['ordinal_refs'].values[0]])
            if np.sum(opinion_dist) == 0: continue
            opinion_dist /= np.sum(opinion_dist)

            refusal_prob = np.sum([dist[v] if v in dist else 0 for v in vdf['refusal_refs'].values[0]])
            refusal_prob /= np.sum(list(dist.values()))

            for kk, vv in zip(['qkey', 'attribute', 'group', 'D_H', 'R_H'],
                              [qkey, demographic, group_name, opinion_dist, refusal_prob]):
                res[kk].append(vv)
        
        
    return pd.DataFrame(res)

def get_max_wd(ordered_ref_weights):
    d0, d1 = np.zeros(len(ordered_ref_weights)), np.zeros(len(ordered_ref_weights))
    d0[np.argmax(ordered_ref_weights)] = 1
    d1[np.argmin(ordered_ref_weights)] = 1
    max_wd = wasserstein_distance(ordered_ref_weights, ordered_ref_weights, d0, d1)
    return max_wd

def get_model_opinions(result_dir, result_files, info_df):
    model_df = []
    for f in result_files:
        context_type = f.split('context=')[1].split(',')[0]
        model_name = f.split('model=')[1].split(',')[0]
        print(f)
        print(model_name, context_type)

        results_json = json.load(open(os.path.join(result_dir, f, 'scenario_state.json'), 'rb'))['request_states']
        mdf = pd.DataFrame([extract_model_opinions(r, context_type, info_df) for r in results_json])

        mdf['results_path'] = f
        mdf['context_type'] = context_type
        mdf['model_name'] = MODEL_NAMES[model_name]
        mdf['model_order'] = MODEL_ORDER[model_name]
        model_df.append(mdf)

        print('-' * 100)
    model_df = pd.concat(model_df)
    return model_df

def get_steering_group(steer_type, steer_df, contexts):
    steer_dict = {}
    for context in contexts:
        if steer_type == 'steer-qa':
            question = context.split('\n')[0].replace('Question: ', '')
            answer_dict = context.split('\n')[1:-1]
            answer_dict = {l.split('. ')[0]: l.split('. ')[1] for l in answer_dict}
            answer = answer_dict[context.split('Answer: ')[1]]
            assert question in steer_df['question'].values
            assert answer in steer_df[steer_df['question'] == question]['correct'].values
            rel = steer_df[np.logical_and(steer_df['question'] == question, 
                                          steer_df['correct'] == answer)]
        else:
            rel = steer_df[steer_df['question'] == context]
        assert len(rel) == 1
        steer_dict[context] = {'attribute': rel['md'].values[0], 
                                   'group': rel['subgroup'].values[0]}
    return steer_dict

VIS_STYLES = [dict(selector="th", props=[('width', '90px'), ("font-size", "95%"),
                                     ('border-left', '1px solid black'), 
                                     ('border-bottom', '1px solid black'), 
                                     ('border-right', '1px solid black'), 
                                     ('border-top', '1px solid black')]),
            dict(selector="td", props=[('text-align', 'center'),
                                       ('border-left', '1px solid black'), 
                                     ('border-bottom', '1px solid black'), 
                                     ('border-right', '1px solid black'), 
                                     ('border-top', '1px solid black')]),
             dict(selector="th.row_heading", props=[('text-align', 'center'), ("font-size", "100%")]),
              dict(selector="th.col_heading",
                   props=[('text-align', 'center'),
                          ('width', '100px'),
                          ('vertical-align', 'top'),
                          ("transform", "translate(0%,10%)"),
                          ("font-size", "70%")
                         ])]