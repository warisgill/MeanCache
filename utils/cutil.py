import torch
import time
from sentence_transformers import util, evaluation
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances

def testDataEvaluation(
    model, sentences1, sentences2, labels, threshold, batch_size):
    def checkCacheHitMiss(emb):
        for j in range(len(embeddings_cached_queries)):
            cosine_score = util.pytorch_cos_sim(emb, embeddings_cached_queries[j])
            if cosine_score >= threshold:
                return j
        return -1

    def cacheHitMiss(res):
        assert len(res) == 1
        id = res[0]["corpus_id"]
        cos_sim = res[0]["score"]
        if cos_sim >= threshold:
            return id
        else:
            return -1

    true_hit = 0
    false_hit = 0
    true_miss = 0
    false_miss = 0

    start_time = time.time()
    model.eval()
    sentences = list(set(sentences1 + sentences2))
    embeddings = model.encode(sentences, batch_size=batch_size, convert_to_tensor=True)
    emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    new_queries = [emb_dict[sent] for sent in sentences1]
    embeddings_cached_queries = [emb_dict[sent] for sent in sentences2]

    resutls = util.semantic_search(
        query_embeddings=new_queries,
        corpus_embeddings=embeddings_cached_queries,
        top_k=1,
    )

    predicted_labels = []
    scores = []
    for i in range(len(labels)):
        cosine_score = util.pytorch_cos_sim(
            new_queries[i], embeddings_cached_queries[i]
        )
        scores.append(cosine_score)
        if cosine_score >= threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

        # if cache_hit_miss:
        #     # r = checkCacheHitMiss(new_queries[i])

        r = cacheHitMiss(resutls[i])
        if r == i and labels[i] == 1:
            true_hit += 1
        elif r != -1 and r != i and labels[i] == 1:
            false_hit += 1
        elif r == -1 and labels[i] == 0:
            true_miss += 1
        elif r == -1 and labels[i] == 1:
            false_miss += 1
        elif labels[i] == 0  and r !=  -1:
            false_hit += 1
        else:
            print(f"Error: r={r}, i={i}, labels[i]={labels[i]}, sentences1[i]={sentences1[i]}, sentences2[i]={sentences2[i]} ")


    assert len(labels) == true_hit + false_hit + true_miss + false_miss

    # compute F1 score
    avg_time = (time.time() - start_time) / len(labels)
    

    f1score = metrics.f1_score(labels, predicted_labels)
    precesion = metrics.precision_score(labels, predicted_labels)
    recall = metrics.recall_score(labels, predicted_labels)
    accuracy = metrics.accuracy_score(labels, predicted_labels)

    label1 = labels.count(1)
    label0 = labels.count(0)
    cosine_scores = 1 - paired_cosine_distances([t.detach().cpu().numpy() for t in  new_queries], [t.detach().cpu().numpy() for t in embeddings_cached_queries])

    d = {
        'AP': metrics.average_precision_score(labels, cosine_scores),
        "F1": f1score,
        "Precision": precesion,
        "Recall": recall,
        "Accuracy": accuracy,
        "True Hit Rate": true_hit/label1,
        "False Hit Rate": false_hit/(label0 + label1),
        "True Miss Rate": true_miss/label0,
        "False Miss Rate": false_miss/label1,
        "Threshold": threshold
    }
    return d

def splitToBalancedValTest(df,  size=100):
    df1 = df[df["is_duplicate"] == 1]
    df0 = df[df["is_duplicate"] == 0]

    df0_chunks = np.array_split(df0, 2) 
    df1_chunks = np.array_split(df1, 2)

    val = pd.concat([df1_chunks[0].head(size), df0_chunks[0].head(size)])
    test = pd.concat([df1_chunks[1].head(size), df0_chunks[1].head(size)])
    return val, test

def _balanceData(df):
    df = df.sample(frac=1).reset_index(drop=True)
    df1 = df[df["is_duplicate"] == 1].sample(frac=1).reset_index(drop=True)
    df0 = df[df["is_duplicate"] == 0].sample(frac=1).reset_index(drop=True)
    size = len(df1)
    val = pd.concat([df1.head(size), df0.head(size)])
    val = val.sample(frac=1).reset_index(drop=True)
    return val

def cacheEvaluator(model, val_data, test_data, batch_size):
    model = model.eval()
    dev_sentences1, dev_sentences2, dev_labels = val_data
    val_df = pd.DataFrame({"question1": dev_sentences1, "question2": dev_sentences2, "is_duplicate": dev_labels})

    val_df= _balanceData(val_df)

    test_df = pd.DataFrame({"question1": test_data[0], "question2": test_data[1], "is_duplicate": test_data[2]})
    test_df = _balanceData(test_df)


    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
        val_df["question1"].tolist(),
        val_df["question2"].tolist(),
        val_df["is_duplicate"].tolist(),
        batch_size=batch_size
    )

    res = binary_acc_evaluator.compute_metrices(model)
    sbert_threshold = res['cossim']['f1_threshold']

    test_data_balanced = test_df['question1'].tolist(), test_df['question2'].tolist(), test_df['is_duplicate'].tolist()


    d = testDataEvaluation(
        model, *test_data, sbert_threshold, batch_size)

    return {"test": d}
