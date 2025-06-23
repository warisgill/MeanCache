# from sentence_transformers import util
from sentence_transformers import SentenceTransformer, models
import torch
from sklearn.metrics.pairwise import paired_cosine_distances


import time
from sentence_transformers import util
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import numpy as np


# from .angle_llama2 import getAngleLlama2
# from .llama2 import llama2Embedding




def TupleToDFDataset(ts1, ts2, tl):
    df = (
        pd.DataFrame({"question1": ts1, "question2": ts2, "is_duplicate": tl})
        .sample(frac=1)
        .reset_index(drop=True)
    )
    return df


def dfToTupleDataset(df):
    return (
        df["question1"].tolist(),
        df["question2"].tolist(),
        df["is_duplicate"].tolist(),
    )


def getEvalMetrics(labels, predicted_labels):
    f_beta_weighted = metrics.fbeta_score(
        labels, predicted_labels, beta=0.5, average="weighted"
    )
    f_beta_binary = metrics.fbeta_score(labels, predicted_labels, beta=0.5)
    
    f1score_binary = metrics.f1_score(labels, predicted_labels)
    f1score_weighted = metrics.f1_score(labels, predicted_labels, average="weighted")
    
    precesion = metrics.precision_score(labels, predicted_labels)
    recall = metrics.recall_score(labels, predicted_labels)
    accuracy = metrics.accuracy_score(labels, predicted_labels)
    cm = metrics.confusion_matrix(labels, predicted_labels)

    answer = {
        "F Beta (Weighted)": f_beta_weighted,
        "F Beta": f_beta_binary,
        "F1 (Weighted)": f1score_weighted,
        "F1": f1score_binary,
        "Precision": precesion,
        "Recall": recall,
        "Accuracy": accuracy,
        "confusion_matrix": cm,
    }

    return answer


def _calculateCacheHitMiss(
    new_q_embeddings,
    c_emeddings,
    sentences1,
    sentences2,
    labels,
    threshold,
    llama2_embeddings=False,
):
    def checkCacheHitMiss(qui):
        for j in range(len(labels)):
            cosine_score = all_scores_cache[qui][j]
            if cosine_score >= threshold:
                return j, cosine_score
        return -1, -1

    all_scores_cache = []

    if llama2_embeddings:
        all_scores_cache = util.cos_sim(new_q_embeddings, c_emeddings)
    else:
        all_scores_cache = util.cos_sim(
            [l.tolist() for l in new_q_embeddings], [l.tolist() for l in c_emeddings]
        )

    true_hit = 0
    false_hit = 0
    true_miss = 0
    false_miss = 0

    hit = 0
    miss = 0
    acc = 0
    # cache hit miss
    total_cached_1 = 0
    store_false_miss = []
    store_false_hit = []
    for i in tqdm(range(len(labels))):
        r, score = checkCacheHitMiss(i)
        # r, score = cacheHitMiss(resutls[i])
        if r == i and labels[i] == 1:
            hit += 1
            total_cached_1 += 1
            acc += 1
        elif labels[i] == 1:
            miss += 1
            total_cached_1 += 1
        elif labels[i] == 0 and r == -1:
            acc += 1

        if r == i and labels[i] == 1:
            true_hit += 1
        elif r != -1 and r != i and labels[i] == 1:
            false_hit += 1
        elif r == -1 and labels[i] == 0:
            true_miss += 1
        elif r == -1 and labels[i] == 1:
            false_miss += 1
            store_false_miss.append(
                {
                    "q": sentences1[i],
                    "cache": sentences2[i],
                    "label": labels[i],
                    "cos_sim": score,
                    "r": r,
                    "i": i,
                }
            )
        elif labels[i] == 0 and r != -1:
            false_hit += 1
            store_false_hit.append(
                {
                    "q": sentences1[i],
                    "cache": sentences2[i],
                    "label": labels[i],
                    "cos_sim": score,
                    "r": r,
                    "i": i,
                }
            )
        else:
            # print(
            #     f"Error: r={r}, i={i}, labels[i]={labels[i]}, sentences1[i]={sentences1[i]}, sentences2[i]={sentences2[i] } "
            # )
            raise Exception("Error in cache hit miss")

    assert len(labels) == true_hit + false_hit + true_miss + false_miss

    label1 = labels.count(1)
    label0 = labels.count(0)

    d1 = {
        "True Hit Rate": true_hit / label1,
        "False Hit Rate": false_hit
        / (label0 + label1),  # important as all misses hit and hits to wrong index
        "True Miss Rate": true_miss / label0,
        "False Miss Rate": false_miss / label1,
    }

    d2 = {
        "Hit Rate": hit / total_cached_1,
        "Miss Rate": miss / total_cached_1,
        "Hit Accuracy": acc / len(labels),
    }

    d2.update(d1)
    # print(
    #     f"sum of rates: {d['True Hit Rate'] + d['False Hit Rate'] + d['True Miss Rate'] + d['False Miss Rate']}"
    # )
    # assert d['True Hit Rate'] + d['False Hit Rate'] + d['True Miss Rate'] + d['False Miss Rate'] == 1.0

    df_false_hit = pd.DataFrame(store_false_hit)
    df_false_hit.to_csv("false_hit.csv")
    df_false_miss = pd.DataFrame(store_false_miss)
    df_false_miss.to_csv("false_miss.csv")

    return d2


def evaluateTransformerModel(
    model, sentences1, sentences2, labels, threshold, hit_miss=True, use_llama2=False
):
    start_time = time.time()
    sentences = list(set(sentences1 + sentences2))

    embeddings = []

    if use_llama2 and model is None:
        print("using llama2 embeddings")
        # embeddings = [model.get_text_embedding(sent) for sent in sentences]
        angle_model = getAngleLlama2()
        # for sen in tqdm(sentences):
        # embeddings.append(model.get_text_embedding(sen))
        # embeddings.append(model.encode(sen, convert_to_tensor=True))
        # angle_model.eval()
        # angle_model = angle_model.to('cuda')
        # embeddings = angle_model.encode([t for t in sentences], to_numpy=True)

        for t in tqdm(sentences):
            # assert len(t) == 1024
            embeddings.append(angle_model.encode(t, to_numpy=True)[0])

        for sen in tqdm(sentences):
            embeddings.append(llama2Embedding(sen))

    else:
        model.eval()
        print("using sbert embeddings")
        embeddings = model.encode(sentences, batch_size=128, convert_to_tensor=True)

    emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    new_queries = [emb_dict[sent] for sent in sentences1]
    embeddings_cached_queries = [emb_dict[sent] for sent in sentences2]

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
            new_q_embeddings=new_queries,
            c_emeddings=embeddings_cached_queries,
            sentences1=sentences1,
            sentences2=sentences2,
            labels=labels,
            threshold=threshold,
            llama2_embeddings=use_llama2,
        )

    cosine_scores = []
    if use_llama2:
        cosine_scores = 1 - paired_cosine_distances(
            new_queries, embeddings_cached_queries
        )
    else:
        cosine_scores = 1 - paired_cosine_distances(
            [t.detach().cpu().numpy() for t in new_queries],
            [t.detach().cpu().numpy() for t in embeddings_cached_queries],
        )

    d = {
        "AP": metrics.average_precision_score(labels, cosine_scores),
        "F1": f1score,
        "Precision": precesion,
        "Recall": recall,
        "Accuracy": accuracy,
        "Threshold": threshold,
        "Avg. Time": avg_time,
    }
    d.update(d_hit_miss)  # add hit miss to d
    return d
