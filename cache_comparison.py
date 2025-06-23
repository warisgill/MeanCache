

from calendar import c
import pandas as pd

from datetime import datetime
import hydra
from utils.dataset import load_datasets

import json

import logging
from tqdm import tqdm
from utils.eutil import (
    TupleToDFDataset,
    dfToTupleDataset,
    getEvalMetrics,
)

from utils.caches import GPTCache, MCacheCTX, GPTCacheCTX, FedGPTCache, FedGPTCacheCompression, LLama2Service




def _expand_queries(queries, labels):
    expanded_queries = []
    i = 0
    for j,q in enumerate(queries):
        cached_entry_child = None
        if'u1_duplicate' in q:
            cached_entry_parent = {'index':i,  'query': q['u0_duplicate'] ,'parent_index':-1, 'metadata': q, 'label': labels[j]}
            expanded_queries.append(cached_entry_parent)
            cached_entry_child = {'index':i+1,  'query': q['u1_duplicate'], 'parent_index':i, 'metadata': q, 'label': labels[j]}
            expanded_queries.append(cached_entry_child)


        elif 'u1_duplicate' not in q and 'u1' in q:
            cached_entry_parent = {'index':i,  'query': q['u0'] ,'parent_index':-1, 'metadata': q, 'label': labels[j]}
            cached_entry_child = {'index':i+1,  'query': q['u1'], 'parent_index':-2, 'metadata': q, 'label': labels[j]} # not a duplicate query so parent index is -2 in the current session
            expanded_queries.append(cached_entry_parent)
            expanded_queries.append(cached_entry_child)
        
        
        i += 2 # increment by 2 is important

    # for q in non_dup_queries:
    #     cached_entry = {'index':i,  'query': q['u0'], 'parent_index':-1, 'metadata': q, 'label': 0}
    #     i += 1
    #     expanded_queries.append(cached_entry)
    # logging.info('Expanded queries: %s', len(expanded_queries))
    return expanded_queries



def eval_cache_performance(llama2_service, new_dup_user_queries, labels):
    response_time = []
    expanded_queries = _expand_queries(new_dup_user_queries, labels)
    true_labels = []
    for q_dict in tqdm(expanded_queries):
        args_dict = {'query': q_dict['query'], 'true_index': q_dict['index'], 'true_label': q_dict['label'], 'parent_id': q_dict['parent_index']} 
        true_labels.append(q_dict['label'])       
        r, t = llama2_service.send_query(args_dict)
        response_time.append(t)
    
    pred_labels = llama2_service.getPredictedLabels()
    return {'time': response_time, 'predicted_labels': pred_labels, 'true_labels': true_labels}

def evaluate1():
    
    _, _, _, server_data = load_datasets("dgptcache", 2, 128)
    val_data, test_data = server_data
    df = TupleToDFDataset(*test_data)

    percent_sim = 0.3
    total_eval_size = 1000

    # sample similalr queries
    df1 = df[df["is_duplicate"] == 1].head(int(percent_sim * total_eval_size))
    df0 = df[df["is_duplicate"] == 0].head(
        total_eval_size - int(percent_sim * total_eval_size)
    )

    df_test = pd.concat([df0, df1]).reset_index(drop=True)

    print(f"Count Values of df_test: {df_test['is_duplicate'].value_counts()}")

    new_queries2, cached_queries, labels = dfToTupleDataset(df_test)

    # new_queries2 = [
    #     "Tell me about computers.",
    #     "Where is new york?",
    #     "What is the capital of France?",
    #     " i love sweet",
    # ]
    # cached_queries = [
    #     "Tell me about computers.",
    #     "how are you",
    #     "What is the capital of France?",
    #     "tell me about dogs",
    # ]
    # labels = [1, 0, 0, 1]

    # total_eval_size = 1000
    currentTime_GMT = f"{datetime.now().timestamp()}"
    currentTime_GMT = currentTime_GMT.split(".")[0]

    llama2_res_times, _ = evalQueries(llama2_service=LLama2Service(None))
    

    # -------------------- Cachesh eval  --------------------------

    llama2_service_with_gptcache_times, gptc_pred_labels = evalQueries(
        llama2_service=LLama2Service(GPTCache(cached_queries), llama2_res_times)
    )

    gptc_eval_dict = getEvalMetrics(labels, gptc_pred_labels)
    gptc_eval_dict["Avg. Time"] = sum(llama2_service_with_gptcache_times) / len(labels)
    gptc_eval_dict["Config"] = "Llama2 + GPTCache"



    mpnet_key = "15th:tname-multi-qa-mpnet-base-cos-v1-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"
    llama2_service_with_fedgptcache_times, fed_pred_labels = evalQueries(
        llama2_service=LLama2Service(FedGPTCache(cached_queries, key=mpnet_key, optimal_threshold= 0.83), llama2_res_times)
    )

    fed_eval_dict = getEvalMetrics(labels, fed_pred_labels)
    fed_eval_dict["Avg. Time"] = sum(llama2_service_with_fedgptcache_times) / len(labels)
    fed_eval_dict["Config"] = "Llama2 + FedGptCache"


    # Albert Evaluation 
    albert_key = "15th:tname-paraphrase-albert-small-v2-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"
    albert_llama2_service_with_fedgptcache_times, albert_fed_pred_labels = evalQueries(
        llama2_service=LLama2Service(FedGPTCache(cached_queries, key=albert_key, optimal_threshold= 0.78), llama2_res_times)
    )
    albert_fed_eval_dict = getEvalMetrics(labels, albert_fed_pred_labels)
    albert_fed_eval_dict["Avg. Time"] = sum(albert_llama2_service_with_fedgptcache_times) / len(labels)
    albert_fed_eval_dict["Config"] = "Llama2 + FedGptCache-Albert"

    

    df_times = pd.DataFrame({
        "Llama2": llama2_res_times,
        "Llama2 + GPTCache": llama2_service_with_gptcache_times,
        "Llama2 + FedGPTCache": llama2_service_with_fedgptcache_times,
        "Llama2 + FedGPTCache (albert)": albert_llama2_service_with_fedgptcache_times,
        
        "GPTCache-Predicted": gptc_pred_labels,
        "FedGptCache-Predicted": fed_pred_labels,
        "FedGptCache-Predicted (albert)": albert_fed_pred_labels,
        "Actual Labels": labels,
    })
    df_times["Sampling"] = f"{df_test.value_counts('is_duplicate')}"
    df_times.to_csv(
        f"csvs/llama2_times_end_to_end_with_dup_query_percent_{percent_sim}_{currentTime_GMT}.csv"
    )

    all_dict = []
    all_dict.append(
        {
            "Config": "Llama2",
            "Avg. Time": sum(llama2_res_times) / len(llama2_res_times),
        }
    )
    all_dict.append(gptc_eval_dict)
    all_dict.append(fed_eval_dict)
    all_dict.append(albert_fed_eval_dict)

    df_metrics = pd.DataFrame(all_dict)
    df_metrics["Sampling"] = f"{df_test.value_counts('is_duplicate')}"
    df_metrics.to_csv(
        f"csvs/llama2_metrics_end_to_end_with_dup_query_percent_{percent_sim}_{currentTime_GMT}.csv"
    )




def _get_non_dup_queries():
    all_queries_u0 = load_dataset('dataset_context/unique_u0.json')
    all_queries_u1 = load_dataset('dataset_context/unique_u1.json')
    all_queries = [{'u0':all_queries_u0[i], 'u1':all_queries_u1[i]} for i in range(len(all_queries_u0))]
    all_queries =  add_index(drop_duplicates(all_queries))
    labels = [0 for _ in range(len(all_queries))]
    return all_queries, labels

def drop_duplicates(all_queries):

    df = pd.DataFrame(all_queries)

    logging.info('Before removing duplicates: %s', df.shape)


    df.drop_duplicates(subset=['u0'], inplace=True)
    logging.info('After removing duplicates: %s', df.shape)

    queries = df.to_dict(orient='records')
    return queries


def load_dataset(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)


def add_index(queries):
    for i, q in enumerate(queries):
        q['index'] = i
    return queries




def eval_cnxt(cfg):
    all_queries = [] 
    for json_file in cfg.json_files:
        all_queries.extend(load_dataset(json_file))
    
    new_queries2 = add_index(drop_duplicates(all_queries))[:75]
    cached_queries = add_index(drop_duplicates(all_queries))[:75]
    all_labels =  [1 for _ in range(len(new_queries2))]

    assert len(new_queries2) == 75
    assert len(cached_queries) == 75
    assert len(all_labels) == 75


    non_dup_queries, temp_labels = _get_non_dup_queries()
    assert len(non_dup_queries) == 75
    cached_queries.extend(non_dup_queries[:25])
    all_labels.extend(temp_labels[:25])

    new_queries2.extend(non_dup_queries[25:])
    all_labels.extend(temp_labels[25:])

    logging.info('Length of cached_queries: %s', len(cached_queries))
    logging.info('Length of new_queries2: %s', len(new_queries2))
    logging.info('Length of all_labels: %s', len(all_labels))







    
    temp_dict = eval_cache_performance(llama2_service=LLama2Service(None), new_dup_user_queries=new_queries2, labels=all_labels)
    llama2_res_times = temp_dict['time']

    # -------------------- Cache Eval  --------------------------
    gpt_c =GPTCacheCTX(cfg.embedding_model_name, cached_queries, cossim_t=cfg.cossim_t, top_k=cfg.top_k)
    dict_gptc = eval_cache_performance(llama2_service=LLama2Service(gpt_c, prev_times=llama2_res_times), new_dup_user_queries=new_queries2, labels=all_labels)
    llama2_service_with_gptcache_times, gptc_pred_labels, true_labels = dict_gptc['time'], dict_gptc['predicted_labels'], dict_gptc['true_labels']
    
    
    
    gptc_eval_dict = getEvalMetrics(true_labels, gptc_pred_labels)
    gptc_eval_dict["Avg. Time"] = sum(llama2_service_with_gptcache_times) / len(true_labels)
    gptc_eval_dict["Config"] = "Llama2 + GPTCache"



    mean_cache = MCacheCTX(cfg.embedding_model_name,cached_queries,  cossim_t=cfg.cossim_t, top_k=cfg.top_k)
    dict_mcache    = eval_cache_performance(llama2_service=LLama2Service(mean_cache, prev_times=llama2_res_times), new_dup_user_queries=new_queries2, labels=all_labels)

    llama2_service_with_fedgptcache_times, fed_pred_labels, true_labels = dict_mcache['time'], dict_mcache['predicted_labels'], dict_mcache['true_labels']
    fed_eval_dict = getEvalMetrics(true_labels, fed_pred_labels)
    fed_eval_dict["Avg. Time"] = sum(llama2_service_with_fedgptcache_times) / len(true_labels)
    fed_eval_dict["Config"] = "Llama2 + MeanCache"


    df_times = pd.DataFrame({
        # "Llama2": llama2_res_times,
        "Llama2 + GPTCache": llama2_service_with_gptcache_times,
        "Llama2 + FedGPTCache": llama2_service_with_fedgptcache_times,
        # "Llama2 + FedGPTCache (albert)": albert_llama2_service_with_fedgptcache_times,
        
        "GPTCache-Predicted-Context": gptc_pred_labels,
        "FedGptCache-Predicted-Context": fed_pred_labels,
        # "FedGptCache-Predicted (albert)": albert_fed_pred_labels,
        "Actual Labels": true_labels,
    })


    # df_times["Sampling"] = f"{df_test.value_counts('is_duplicate')}"
    df_times.to_csv(
        f"csvs/llama2_times_end_to_end_with_dup_query_percent_meancache.csv"
    )

    all_dict = []
    # all_dict.append(
    #     {
    #         "Config": "Llama2",
    #         "Avg. Time": sum(llama2_res_times) / len(llama2_res_times),
    #     }
    # )
    all_dict.append(gptc_eval_dict)
    all_dict.append(fed_eval_dict)
    # all_dict.append(albert_fed_eval_dict)

    df_metrics = pd.DataFrame(all_dict)
    # df_metrics["Sampling"] = f"{df_test.value_counts('is_duplicate')}"
    df_metrics.to_csv(f"csvs/contextual_queries_{cfg.embedding_model_name}.csv")
    return df_metrics


# cossim_t: 0.5
# top_k: 5

def eval_differnt_thresholds_topks(cfg):
    all_dfs = []

    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for top_k in [5]:
            cfg.cossim_t = t
            cfg.top_k = top_k
            df = eval_cnxt(cfg)
            all_dfs.append((t, top_k, df))
    
    print("Overall Evaluations ")

    for t, top_k, df in all_dfs:
        print(f"--------> Threshold: {t}, Top_k: {top_k}")
        print(df)

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg):
    
    df = eval_cnxt(cfg)
    print(df) 


if __name__ == "__main__":    
    main()
