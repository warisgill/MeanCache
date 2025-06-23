import pandas as pd
from utils.dataset import load_datasets
from sentence_transformers import SentenceTransformer
from diskcache import Index
import numpy as np
from utils.eutil import evaluateTransformerModel
from utils.threshold_opt import findThreshold
from datetime import datetime
import time
from tqdm import tqdm


global_model_cache = Index(".storage/cache/global_models")
rounds_cache = Index(".storage/cache/rounds")


def _reduceData(df, size):
    df = df.sample(frac=1).reset_index(drop=True)
    df1 = df[df["is_duplicate"] == 1].sample(frac=1).reset_index(drop=True)
    df0 = df[df["is_duplicate"] == 0].sample(frac=1).reset_index(drop=True)
    val = pd.concat([df1.head(size), df0.head(size)])
    val = val.sample(frac=1).reset_index(drop=True)
    return val


def gptCacheEvalution(test_data, tname):
    model = SentenceTransformer(tname)
    model.eval()
    model = model.to("cuda")
    r = evaluateTransformerModel(
        model,
        *test_data,
        0.7,
        hit_miss=True,
    )
    model = model.to("cpu")
    return r


def fedGPTCacheEvaluation(key, val_data, test_data):
    model = global_model_cache[key][0]
    model.eval()
    model = model.to("cuda")
    # fining the best threshold from validation set
    print("> Finding the best threshold from validation set")
    sbert_threshold = findThreshold(model, val_data, hit_miss=False)

    result2 = evaluateTransformerModel(
        model,
        *test_data,
        sbert_threshold,
        hit_miss=True,
    )

    # result1 = {f"{k} (My-Threshold)": v for k, v in result1.items()}
    result = {f"{k} (SBERT-Threshold)": v for k, v in result2.items()}

    # result = {**result1, **result2}

    model = model.to("cpu")
    return result


def singleKeyTest(select_key, val_data, test_data):
    def _foo():
        final_roundf1 = -1
        stored_best_f1 = -1
        rounds_f1 = []
        rounds_ap = []
        rounds_precsion = []
        rounds_true_hit = []
        rounds_true_miss = []
        rounds_acc = []

        for k2, v2 in round2results.items():
            if "F1" not in v2["test"]:
                return {}
            rounds_f1.append(v2["test"]["F1"])
            if "AP" in v2["test"]:
                rounds_ap.append(v2["test"]["AP"])
            rounds_precsion.append(v2["test"]["Precision"])
            if "True Hit Rate" in v2["test"]:
                rounds_true_hit.append(v2["test"]["True Hit Rate"])
                rounds_true_miss.append(v2["test"]["True Miss Rate"])
                rounds_acc.append(v2["test"]["Accuracy"])

            if "Best F1" in v2:
                stored_best_f1 = v2["Best F1"]

            final_roundf1 = v2["test"]["F1"]

        return {
            "F1 (Final Round)": final_roundf1,
            "F1 (Stored Best F1)": stored_best_f1,
            "F1 (All Rounds)": rounds_f1,
            "AP (All Rounds)": rounds_ap,
            "Precision (All Rounds)": rounds_precsion,
            # "True Hit Rate (All Rounds)": rounds_true_hit,
            # "True Miss Rate (All Rounds)": rounds_true_miss,
            # "Accuracy (All Rounds)": rounds_acc,
        }

    config = rounds_cache[select_key][0][0]
    round2results = rounds_cache[select_key][0][1]
    log_dict = _foo()

    config.update(log_dict)
    d = fedGPTCacheEvaluation(select_key, val_data, test_data)
    d = {f"{k}": v for k, v in d.items()}
    config.update(d)
    config["key"] = select_key

    return config


def allCacheKeysCompleteEval(dname):
    _, _, _, server_data = load_datasets(dname, 2, 128)
    val_data, test_data = server_data
    vs1, vs2, vl = val_data

    df_val = (
        pd.DataFrame({"question1": vs1, "question2": vs2, "is_duplicate": vl})
        .sample(frac=1)
        .reset_index(drop=True)
    )
    df_val = _reduceData(df_val, 1000)

    ts1, ts2, tl = test_data
    df_test = (
        pd.DataFrame({"question1": ts1, "question2": ts2, "is_duplicate": tl})
        .sample(frac=1)
        .reset_index(drop=True)
    )
    df_test = _reduceData(df_test, 1000)

    val_data = (
        df_val["question1"].tolist(),
        df_val["question2"].tolist(),
        df_val["is_duplicate"].tolist(),
    )
    test_data = (
        df_test["question1"].tolist(),
        df_test["question2"].tolist(),
        df_test["is_duplicate"].tolist(),
    )

    all_dict = []

    for k in global_model_cache:
        if k.find(dname) == -1:
            continue
        print(f"> Evaluating {k}")
        r_dict = singleKeyTest(k, val_data, test_data)
        gpt_cache_dict = gptCacheEvalution(
            test_data, r_dict["tname"]
        )  # to store corrsponding gptcache result
        gpt_cache_dict = {f"{k} (GPTCache)": v for k, v in gpt_cache_dict.items()}
        r_dict.update(gpt_cache_dict)
        r_dict = dict(sorted(r_dict.items()))
        all_dict.append(r_dict)

    df = pd.DataFrame(all_dict)
    currentTime_GMT = f"{datetime.now().timestamp()}"
    currentTime_GMT = currentTime_GMT.split(".")[0]
    total = len(test_data[0])
    labels = test_data[2]
    values, counts = np.unique(labels, return_counts=True)
    df["Sampling"] = f"{values}:{counts}, Total:{total}"
    df.to_csv(
        f"csvs/a1_fedcache_all_{dname}{currentTime_GMT}.csv",
        index=False,
    )


def csvSelectedCacheKeysCompleteEval(dname, select_keys):
    _, _, _, server_data = load_datasets(dname, 2, 128)
    val_data, test_data = server_data
    vs1, vs2, vl = val_data

    df_val = (
        pd.DataFrame({"question1": vs1, "question2": vs2, "is_duplicate": vl})
        .sample(frac=1)
        .reset_index(drop=True)
    )

    ts1, ts2, tl = test_data
    df_test = (
        pd.DataFrame({"question1": ts1, "question2": ts2, "is_duplicate": tl})
        .sample(frac=1)
        .reset_index(drop=True)
    )
    
    # df_test = _reduceData(df_test, 2000)
    df_val = _reduceData(df_val, 2000)
    df_test = _reduceData(df_test, 2000)


    val_data = (
        df_val["question1"].tolist(),
        df_val["question2"].tolist(),
        df_val["is_duplicate"].tolist(),
    )
    test_data = (
        df_test["question1"].tolist(),
        df_test["question2"].tolist(),
        df_test["is_duplicate"].tolist(),
    )

    all_dfs = []
    
    # # llamma2 evaluation
    # llamma2_model = getLLAMMA2Model()
    # d =  evaluateTransformerModel(llamma2_model, *test_data, 0.7, hit_miss=True, use_llama2=True)
    # d['Cache Type'] = 'GPTCache-llama2'
    # d['Model'] = 'llama2'
    # d['Key'] = '---'
    # all_dict.append(d)

    
    
    d = gptCacheEvalution(test_data, "paraphrase-albert-small-v2")

    gptcache_dict = {
        'Metric': list(d.keys()),
        'Value': list(d.values()),
    }
    df_gptcache = pd.DataFrame(gptcache_dict)
    df_gptcache["Cache Type"] = "GPTCache"
    df_gptcache["Model"] = "paraphrase-albert-small-v2"
    df_gptcache["Key"] = "---"

    all_dfs.append(df_gptcache)

    
    
    # gpt_cache_dict["Cache Type"] = "GPTCache"
    # gpt_cache_dict["Model"] = "paraphrase-albert-small-v2"
    # gpt_cache_dict["Key"] = "---"
    # all_dict.append(gpt_cache_dict)

    


    for k in select_keys:
        print(f"> Evaluating {k}")
        # r_dict = singleKeyTest(k, val_data, test_data)
        # d = fedGPTCacheEvaluation(k, val_data, test_data)
        d = {k.replace(' (SBERT-Threshold)', ''): v for k, v in d.items()}
        fedgptcache_dict = {'Metric': list(d.keys()), 'Value': list(d.values())}
        df_fedgptcache = pd.DataFrame(fedgptcache_dict)
        
        if k.find("mpnet") != -1:
            df_fedgptcache["Cache Type"] = "FedGPTCache-mpnet"
            df_fedgptcache["Model"] = 'mpnet'
        elif k.find("albert") != -1:
            df_fedgptcache["Cache Type"] = "FedGPTCache-albert"
            df_fedgptcache["Model"] = 'albert'
        else:
            raise Exception("Unknown model FedGPTCache")
        
        df_fedgptcache['Key'] = k

        all_dfs.append(df_fedgptcache)

    # df = pd.DataFrame(all_dict)
        
    df = pd.concat(all_dfs)
    
    currentTime_GMT = f"{datetime.now().timestamp()}"
    currentTime_GMT = currentTime_GMT.split(".")[0]
    total = len(test_data[0])
    labels = test_data[2]
    values, counts = np.unique(labels, return_counts=True)
    df["Sampling"] = f"{values}:{counts}, Total:{total}"

    df.to_csv(
        f"csvs/plot_FedGPTCache-vs-GPTCache-{dname}{currentTime_GMT}.csv",
        index=False,
    )


def csvFLTrain(key):
    round2results = rounds_cache[key][0][1]

    # print(round2results)

    all_results = []
    for k, v in round2results.items():
        all_results.append(v["test"])

    df = pd.DataFrame(all_results)
    df["Key"] = key

    c = rounds_cache[key][0][0]
    main_exp_info = f"{c['dname']}_{c['tname']}_{c['num_clients']}_{c['num_rounds']}"
    fname = f"csvs/plot_fl_training_{main_exp_info}.csv"
    df.to_csv(fname, index=False)
    print(f"> Training saved (R2 Metric) {fname}")
    return None



def csvThreshold(dname, select_keys):
    _, _, _, server_data = load_datasets(dname, 2, 128)
    val_data, test_data = server_data
    ts1, ts2, tl = test_data
    df_test = (
        pd.DataFrame({"question1": ts1, "question2": ts2, "is_duplicate": tl})
        .sample(frac=1)
        .reset_index(drop=True)
    )
    
    df_test = _reduceData(df_test, 100)


    test_data = (
        df_test["question1"].tolist(),
        df_test["question2"].tolist(),
        df_test["is_duplicate"].tolist(),
    )

    all_dict= []

    
    # llamma2 evaluation
    llamma2_model = None
    for t in tqdm([i/100 for i in range(0, 100)]):
        d =  evaluateTransformerModel(llamma2_model, *test_data, t, hit_miss=False, use_llama2=True)
        d = {k.replace(' (SBERT-Threshold)', ''): v for k, v in d.items()}
        d['Cache Type'] = 'llama2'
        d['Model'] = 'llama2'
        d['Key'] = '---'
        all_dict.append(d)

        for k in select_keys:
            print(f"> Evaluating {k}")
            model = global_model_cache[k][0]
            model.eval()
            model = model.to("cuda")

            d = evaluateTransformerModel(model, *test_data, t, hit_miss=False,)
            d = {k.replace(' (SBERT-Threshold)', ''): v for k, v in d.items()}            
            
            if k.find("mpnet") != -1:
                d["Cache Type"] = "FedGPTCache-mpnet"
                d["Model"] = 'mpnet'
            elif k.find("albert") != -1:
                d["Cache Type"] = "FedGPTCache-albert"
                d["Model"] = 'albert'
            else:
                raise Exception("Unknown model FedGPTCache")
            
            d['Key'] = k

            all_dict.append(d)
            model = model.to("cpu")

    df = pd.DataFrame(all_dict) 
    currentTime_GMT = f"{datetime.now().timestamp()}"
    currentTime_GMT = currentTime_GMT.split(".")[0]
    total = len(test_data[0])
    labels = test_data[2]
    values, counts = np.unique(labels, return_counts=True)
    df["Sampling"] = f"{values}:{counts}, Total:{total}"

    df.to_csv(
            f"csvs/plot_threshold_variation-{dname}{currentTime_GMT}.csv",
            index=False,
        )






if __name__ == "__main__":
    dname = "dgptcache"
    _, _, _, server_data = load_datasets(dname, 2, 128)
    val_data, test_data = server_data
    # allCacheKeysCompleteEval("quora")

    key1 = "15th:tname-paraphrase-albert-small-v2-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"
    key2 = "15th:tname-multi-qa-mpnet-base-cos-v1-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"

    # csvFLTrain(key1)
    # csvFLTrain(key2)
    # csvSelectedCacheKeysCompleteEval("dgptcache", [key1, key2])
    # csvThreshold('dgptcache', [key1, key2])

    
