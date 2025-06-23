import pandas as pd
from utils.dataset import _loadGPTCacheDataset, _loadQuoraDataset

# from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from diskcache import Index

import time
from sentence_transformers import util
from sklearn import metrics

import numpy as np

from utils.eutil import evaluateTransformerModel
from utils.threshold_opt import findThreshold


global_model_cache = Index(".storage/cache/global_models")
rounds_cache = Index(".storage/cache/rounds")


def _reduceData(df, size):
    df = df.sample(frac=1).reset_index(drop=True)
    df1 = df[df["is_duplicate"] == 1].sample(frac=1).reset_index(drop=True)
    df0 = df[df["is_duplicate"] == 0].sample(frac=1).reset_index(drop=True)
    val = pd.concat([df1.head(size), df0.head(size)])
    val = val.sample(frac=1).reset_index(drop=True)
    return val


def _splitToBalancedValTest(df):
    df1 = df[df["is_duplicate"] == 1]
    df0 = df[df["is_duplicate"] == 0]

    df0_chunks = np.array_split(df0, 2)
    df1_chunks = np.array_split(df1, 2)

    val = pd.concat([df1_chunks[0], df0_chunks[0]])
    test = pd.concat([df1_chunks[1], df0_chunks[1]])
    val = val.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    return val, test


def loaValTest(dname):
    all_test = None
    if dname == "dgptcache":
        _, _, all_test = _loadGPTCacheDataset()
    elif dname == "quora":
        _, _, all_test = _loadQuoraDataset()

    print(f"> Dataset {dname} Loaded")

    # shuffle the df
    all_test = all_test.sample(frac=1).reset_index(drop=True)

    df_val, df_test = _splitToBalancedValTest(all_test)

    df_val = _reduceData(df_val, 1000)  # to make evaluation faster

    return df_val, df_test


# def allCacheKeysPrint():
#     for k in global_model_cache:
#         print(k)


def generateEvaluationCSVs(select_key, tname, df_val, df_test):
    def _fedCacheEvaluation(df_val, df_test):
        model = global_model_cache[select_key][0]
        model.eval()
        model = model.to("cuda")
        # fining the best threshold from validation set
        print("> Finding the best threshold from validation set")
        best_threshold = findThreshold(model, df=df_val)

        # evaluating the model on test set
        result = evaluateTransformerModel(
            model,
            df_test["question1"].tolist(),
            df_test["question2"].tolist(),
            df_test["is_duplicate"].tolist(),
            best_threshold,
            hit_miss=True,
        )
        
        return result

    def _csvGPTCacheFedGPTCache():
        gptcache_result_df = pd.DataFrame(
            {
                "Cache Type": "GPTCache",
                "Metric": gptcache_result.keys(),
                "Value": gptcache_result.values(),
            }
        )
        fedgptcache_result_df = pd.DataFrame(
            {
                "Cache Type": "FedGPTCache",
                "Metric": fedgptcache_result.keys(),
                "Value": fedgptcache_result.values(),
            }
        )

        df = pd.concat([gptcache_result_df, fedgptcache_result_df])
        df["Key"] = select_key
        fname = f"csvs/plot_GPTcache_vs_FedGPTCache_{main_exp_info}.csv"
        df.to_csv(fname, index=False)
        print(f"> GPTCache vs FedGPTCache CSV Saved to {fname}")

    def _csvImpactOfThreshold(thresholds, ts_results):
        df = pd.DataFrame(ts_results)
        df["Threshold"] = thresholds
        df["Key"] = select_key

        fname = f"csvs/plot_threshold_variation_impact_{main_exp_info}.csv"
        df.to_csv(fname, index=False)
        print(f"> Threshold Variation Impact CSV Saved to {fname}")

    # ******** Main ********

    # Gptcache Result
    print("> Evaluating GPTCache")
    gptcache_result = _gptCacheEvalution(df_test)
    print("> gptcacheresult", gptcache_result)

    print("> Evaluating FedGPTCache")
    assert select_key.find(tname) != -1
    fedgptcache_result, all_ts, all_ts_results = _fedCacheEvaluation(df_val, df_test)
    print("> fedgptcacheresult", fedgptcache_result)

    c = rounds_cache[select_key][0][0]
    main_exp_info = f"{c['dname']}_{c['tname']}_{c['num_clients']}_{c['num_rounds']}"
    _csvGPTCacheFedGPTCache()
    _csvImpactOfThreshold(all_ts, all_ts_results)


# generate the csv for embedding drift
def generateEmbeddingDriftCSV(key, df):
    def _evaluateTransformerModelDrift(
        model1, model2, sentences1, sentences2, labels, threshold, hit_miss=False
    ):
        start_time = time.time()

        model1.eval()
        model2.eval()

        new_queries = model1.encode(
            sentences1, batch_size=128, convert_to_tensor=True
        )  # [emb_dict[sent] for sent in sentences1]
        new_queries = new_queries.cpu()
        embeddings_cached_queries = model2.encode(
            sentences2, batch_size=128, convert_to_tensor=True
        )  # [emb_dict[sent] for sent in sentences2]
        embeddings_cached_queries = embeddings_cached_queries.cpu()

        predicted_labels = []
        for i in range(len(labels)):
            cosine_score = util.pytorch_cos_sim(
                new_queries[i], embeddings_cached_queries[i]
            )
            if cosine_score >= threshold:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

        avg_time = (time.time() - start_time) / len(labels)

        f1score = metrics.f1_score(labels, predicted_labels)
        precesion = metrics.precision_score(labels, predicted_labels)
        recall = metrics.recall_score(labels, predicted_labels)
        accuracy = metrics.accuracy_score(labels, predicted_labels)
        d_hit_miss = {}
        if hit_miss:
            print("> Calculating Hit Miss")
            d_hit_miss = _calculateCacheHitMiss(
                new_queries_embeddings=new_queries,
                cached_emeddings=embeddings_cached_queries,
                sentences1=sentences1,
                sentences2=sentences2,
                labels=labels,
                threshold=threshold,
            )

        d = {
            "F1": f1score,
            "Precision": precesion,
            "Recall": recall,
            "Accuracy": accuracy,
            "Threshold": threshold,
            "Avg. Time": avg_time,
        }
        d.update(d_hit_miss)  # add hit miss to d

        return d

    global_model = global_model_cache[key][0]
    global_model.eval()
    global_model = global_model.to("cuda")

    # _, _, threshold =  findThreshold(global_model, df)
    threshold = 0.62

    config = rounds_cache[key][0][0]
    r2result = rounds_cache[key][0][1]
    server_round = config["num_rounds"] - 1

    # prev_round_model = r2result[server_round][f"global_model-r{server_round}"]
    # prev_round_model.eval()
    # prev_round_model = prev_round_model.to("cuda")
    f1_model = global_model_cache[key][1]
    f1_model.eval()
    f1_model = f1_model.to("cuda")

    drift = _evaluateTransformerModelDrift(
        global_model,
        f1_model,
        df["question1"].tolist(),
        df["question2"].tolist(),
        df["is_duplicate"].tolist(),
        threshold,
        hit_miss=True,
    )
    # print(r1)
    same_model = _evaluateTransformerModelDrift(
        global_model,
        global_model,
        df["question1"].tolist(),
        df["question2"].tolist(),
        df["is_duplicate"].tolist(),
        threshold,
        hit_miss=True,
    )

    df_drift = pd.DataFrame(
        {
            "Model": "Different Models",
            "Metric": drift.keys(),
            "Value": drift.values(),
        }
    )
    df_same_model = pd.DataFrame(
        {
            "Model": "Same Model",
            "Metric": same_model.keys(),
            "Value": same_model.values(),
        }
    )

    df = pd.concat([df_drift, df_same_model])

    print(df)

    df["Key"] = key

    c = rounds_cache[key][0][0]
    main_exp_info = f"{c['dname']}_{c['tname']}_{c['num_clients']}_{c['num_rounds']}"
    fname = f"csvs/plot_embedding_drift_{main_exp_info}.csv"
    df.to_csv(fname, index=False)
    print(f"> Embedding Drift CSV Saved to {fname}")


# generate the csv for round 2 metrics
def generateRound2MetricsCSV(key):
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


# compare the model size and compute time for different models
def generateModelSizeOverHeadComparisonCSV(df):
    tnames = [
        "paraphrase-albert-small-v2",
        "paraphrase-MiniLM-L3-v2",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ]
    compute_times = []
    avg_model_sizes = [43, 61, 80, 420]

    # shuffle the df
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.head(1000)

    for tname in tnames:
        model = SentenceTransformer(tname)
        model.eval()
        model = model.cpu()
        start_time = time.time()
        all_queries = df["question1"].tolist() + df["question2"].tolist()
        embeddings = model.encode(all_queries, batch_size=128, convert_to_tensor=True)
        avg_time = time.time() - start_time
        compute_times.append(avg_time)

    plot_df = pd.DataFrame(
        {
            "Model": tnames,
            "Embeddings Compute Time (s)": compute_times,
            "Model Size (MB)": avg_model_sizes,
        }
    )
    plot_df["Avg. Embeddings Compute Time (s)"] = plot_df[
        "Embeddings Compute Time (s)"
    ] / len(df)
    plot_df["Num Queries"] = len(all_queries)
    fname = f"csvs/plot_model_size_overhead_comparison.csv"
    plot_df.to_csv(fname, index=False)
    print(plot_df)
    print(f"> Model Size Overhead Comparison CSV Saved to {fname}")
    return None





def gptCacheEvalution(df_test, tname):
    model = SentenceTransformer(tname)
    model.eval()
    model = model.to("cuda")
    r = evaluateTransformerModel(
        model,
        df_test["question1"].tolist(),
        df_test["question2"].tolist(),
        df_test["is_duplicate"].tolist(),
        0.7,
        hit_miss=True,
    )
    model = model.to("cpu")
    return r


def detailEachKeyTest(select_key, df_val, df_test):
    def _foo():
        best_f1 = -1
        t_round = -1
        f1_key = "F1"
        final_rounf1 = -1
        stored_best_f1 = -1
        true_hit_final_round = -1
        true_hit_round = -1
        for k2, v2 in round2results.items():
            if "f1" in v2["test"]:
                f1_key = "f1"

            if v2["test"][f1_key] > best_f1:
                best_f1 = v2["test"][f1_key]
                t_round = k2
                if "True Hit Rate" in v2["test"]:
                    true_hit_round = v2["test"]["True Hit Rate"]

            if "True Hit Rate" in v2["test"]:
                true_hit_final_round = v2["test"]["True Hit Rate"]

            if "Best F1" in v2:
                stored_best_f1 = v2["Best F1"]

            final_rounf1 = v2["test"][f1_key]
        return (
            best_f1,
            t_round,
            final_rounf1,
            stored_best_f1,
            true_hit_round,
            true_hit_final_round,
        )

    config = rounds_cache[select_key][0][0]
    round2results = rounds_cache[select_key][0][1]
    rounf1, r, final_r_f1, stored_best_f1, t_hit_round, t_hit_final = _foo()
    result_dict = {
        "Searched Round": r,
        "Round F1": rounf1,
        "Final Round F1": final_r_f1,
        "Stored Best F1": stored_best_f1,
        "True Hit Rate (Round)": t_hit_round,
        "True Hit Rate (Final Round)": t_hit_final,
    }

    config.update(result_dict)
    d = fedGPTCacheEvaluation(select_key, df_val, df_test)
    d = {f"{k} (Evaluated)": v for k, v in d.items()}
    config.update(d)
    config["key"] = select_key

    return config


def fedGPTCacheEvaluation(key, df_val, df_test):
    model = global_model_cache[key][0]
    model.eval()
    model = model.to("cuda")
    # fining the best threshold from validation set
    print("> Finding the best threshold from validation set")
    _, _, best_threshold, sbert_threshold = findThreshold(model, df=df_val, hit_miss=False)
    print(f"> Best Threshold: {best_threshold}")
    # evaluating the model on test set
    result1 = evaluateTransformerModel(
        model,
        df_test["question1"].tolist(),
        df_test["question2"].tolist(),
        df_test["is_duplicate"].tolist(),
        best_threshold,
        hit_miss=True,
    )

    result2 = evaluateTransformerModel(
        model,
        df_test["question1"].tolist(),
        df_test["question2"].tolist(),
        df_test["is_duplicate"].tolist(),
        sbert_threshold,
        hit_miss=True,
    )

    result1 = {f"{k} (My-Threshold)": v for k, v in result1.items()}
    result2 = {f"{k} (SBERT-Threshold)": v for k, v in result2.items()}

    result = {**result1, **result2}

    model = model.to("cpu")
    return result


def allCacheKeysCompleteEval(dname):
    df_val, df_test = loaValTest(dname)
    all_dict = []

    for k in global_model_cache:
        if k.find(dname) == -1:
            continue
        print(f"> Evaluating {k}")
        r_dict = detailEachKeyTest(k, df_val, df_test)
        gpt_cache_dict = gptCacheEvalution(df_test, r_dict["tname"]) # to store corrsponding gptcache result
        gpt_cache_dict = {f"{k} (GPTCache)": v for k, v in gpt_cache_dict.items()}
        r_dict.update(gpt_cache_dict)
        all_dict.append(r_dict)
        break

    df = pd.DataFrame(all_dict)
    df = df.sort_values(by=["Round F1"], ascending=False)
    df.to_csv(f"csvs/a1_size_{len(df_test)}fedcache_all_{dname}.csv", index=False)


def _main():
    dname = "dgptcache"
    tname = "paraphrase-albert-small-v2"
    all_test = None
    if dname == "dgptcache":
        _, _, all_test = _loadGPTCacheDataset()
    elif dname == "quora":
        _, _, all_test = _loadQuoraDataset()

    print(f"> Dataset {dname} Loaded")

    df_val, df_test = _splitToBalancedValTest(all_test)

    select_key = "15th:tname-multi-qa-mpnet-base-cos-v1-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"

    assert select_key.find(tname) != -1
    assert select_key.find(dname) != -1

    df_val = _reduceData(df_val, 1000)  # to make evaluation faster
    df_test = _reduceData(df_test, 1000)  # to make evaluation faster

    assert len(df_val) <= 2000
    assert len(df_test) <= 2000

    print(df_test.value_counts("is_duplicate"))

    # generateEvaluationCSVs(select_key, tname, df_val, df_test)
    generateRound2MetricsCSV(select_key)
    # generateModelSizeOverHeadComparisonCSV(df_test)
    # generateEmbeddingDriftCSV(select_key, df_test)


if __name__ == "__main__":
    # allCacheKeysPrint()
    # main()
    # main2("dgptcache")
    # main2("quora")
    allCacheKeysCompleteEval("quora")
    # allCacheKeysCompleteEval("dgptcache")
