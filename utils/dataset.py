import logging
import os
import csv
from zipfile import ZipFile
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import util
from sentence_transformers.readers import InputExample
import numpy as np


def _loadGPTCacheDataset():
    df_train = pd.read_csv("dataset_gptcache/train.csv")
    df_val = pd.read_csv("dataset_gptcache/val.csv")
    df_test = pd.read_csv("dataset_gptcache/test.csv")

    return df_train, df_val, df_test

def _loadQuoraDataset():
    df_train_classificaiton = pd.read_csv(
            ".storage/dataset/quora-IR-dataset/classification/train_pairs.tsv",
            sep="\t",
            quoting=csv.QUOTE_NONE,
            encoding="utf8",
        )
    df_dev_classificaiton = pd.read_csv(
            ".storage/dataset/quora-IR-dataset/classification/dev_pairs.tsv",
            sep="\t",
            quoting=csv.QUOTE_NONE,
            encoding="utf8",
        )

    df_test = pd.read_csv(
            ".storage/dataset/quora-IR-dataset/classification/test_pairs.tsv",
            sep="\t",
            quoting=csv.QUOTE_NONE,
            encoding="utf8",
        )
    
    return df_train_classificaiton, df_dev_classificaiton, df_test






def prepareUniqueTestData(df):
    # df = df[df['label'] == 1]

    print(f"rows : {len(df)}")
    q1 = df["question1"].tolist()
    q2 = df["question2"].tolist()
    qs_all = q1 + q2
    uniq_qs = list(set(qs_all))
    print(f"all quries: {len(qs_all)}")
    print(f"uniqe queries: {len(uniq_qs)}")
    query2id = {sen: i for i, sen in enumerate(uniq_qs)}

    id2Done = {i: False for i, _ in enumerate(uniq_qs)}

    uniqe_rows = []

    # iterat over rows
    for _, row in df.iterrows():
        text_a = row["question1"]
        text_b = row["question2"]
        q1_id = query2id[text_a]
        q2_id = query2id[text_b]

        if id2Done[q1_id] == False and id2Done[q2_id] == False:
            uniqe_rows.append(row)
            id2Done[q1_id] = True
            id2Done[q2_id] = True

    df = pd.DataFrame(uniqe_rows)

    print(f"uniqe rows: {len(df)}")
    return df


def _downloadDataset(dname):
    if dname == "quora":
        dataset_path = ".storage/dataset/quora-IR-dataset"
        if not os.path.exists(dataset_path):
            logging.info("Dataset not found. Download")
            zip_save_path = ".storage/dataset/quora-IR-dataset.zip"
            util.http_get(
                url="https://sbert.net/datasets/quora-IR-dataset.zip",
                path=zip_save_path,
            )
            with ZipFile(zip_save_path, "r") as zip:
                zip.extractall(dataset_path)


def _splitDataframe(df, n):
    chunks = np.array_split(df, n)
    return chunks


def _dfs(dname):
    if dname == "quora":
        return _loadQuoraDataset()
    elif dname == "dgptcache":
        df_train_classificaiton, df_dev_classificaiton, df_test = _loadGPTCacheDataset()
    else:
        # raise exception that dataset not found
        raise Exception("Dataset not found")

    df_train_classificaiton = df_train_classificaiton.sample(frac=1).reset_index(
            drop=True)
    df_dev_classificaiton = df_dev_classificaiton.sample(frac=1).reset_index(
            drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    
    df_train_classificaiton = df_train_classificaiton.dropna()
    df_dev_classificaiton = df_dev_classificaiton.dropna()
    df_test = df_test.dropna()

    return df_train_classificaiton, df_dev_classificaiton, df_test


def _getTrainLoader(df_train_classificaiton, train_batch_size):
    train_samples_MultipleNegativesRankingLoss = []
    train_samples_ConstrativeLoss = []
    # counter = 0
    for _, row in df_train_classificaiton.iterrows():
        train_samples_ConstrativeLoss.append(
            InputExample(
                texts=[row["question1"], row["question2"]],
                label=int(row["is_duplicate"]),
            )
        )
        if row["is_duplicate"] == 1:
            # counter += 1
            train_samples_MultipleNegativesRankingLoss.append(
                InputExample(texts=[row["question1"], row["question2"]], label=1)
            )
            train_samples_MultipleNegativesRankingLoss.append(
                InputExample(texts=[row["question2"], row["question1"]], label=1)
            )  # if A is a duplicate of B, then B is a duplicate of A

        # if counter >= 256:
        #     break
    # Create data loader and loss for MultipleNegativesRankingLoss
    train_dl_mnr = DataLoader(
        train_samples_MultipleNegativesRankingLoss,
        shuffle=True,
        batch_size=train_batch_size,
    )

    # Create data loader and loss for OnlineContrastiveLoss
    train_dl_c = DataLoader(
        train_samples_ConstrativeLoss, shuffle=True, batch_size=train_batch_size
    )

    return (train_dl_mnr, train_dl_c)


def _loadRawData(df_dev):
    ######### Test Data  ##########
    dev_sentences1 = []
    dev_sentences2 = []
    dev_labels = []

    # counter = 0
    for _, row in df_dev.iterrows():
        dev_sentences1.append(row["question1"])
        dev_sentences2.append(row["question2"])
        dev_labels.append(int(row["is_duplicate"]))
        # counter += 1
        # if counter >= 256:
        #     break

    raw_data = (dev_sentences1, dev_sentences2, dev_labels)
    return raw_data


# def _balancedSampling(df, val_test_frac):
#     df_label_1 = df[df["is_duplicate"] == 1]
#     df_label_0 = df[df["is_duplicate"] == 0]


#     df_label_1 = df_label_1.sample(frac=val_test_frac).reset_index(drop=True)
#     df_label_0 = df_label_0.head(len(df_label_1))

#     print(f"> Dataset Distribution Label 1: {len(df_label_1)}, Label 0: {len(df_label_0)}")

#     df = pd.concat([df_label_1, df_label_0])
#     df = df.sample(frac=1).reset_index(drop=True)
#     return df


def _splitValDataBalanced(df, num_clients):
    df1 = df[df["is_duplicate"] == 1]
    df0 = df[df["is_duplicate"] == 0]
    df1_chunks = _splitDataframe(df1, num_clients)
    df0_chunks = _splitDataframe(df0, num_clients)
    dfs = []
    for i in range(num_clients):
        temp_df = pd.concat([df1_chunks[i], df0_chunks[i]])
        temp_df = temp_df.sample(frac=1).reset_index(drop=True)
        dfs.append(temp_df)
    return dfs


def load_datasets(dname, num_clients, train_batch_size):
    ######### Train Data  ##########
    _downloadDataset(dname)
    df_train, df_val_all, df_test_all = _dfs(dname)
    dfs = _splitDataframe(df_train, num_clients)
    train_dls = [_getTrainLoader(df, train_batch_size) for df in dfs]

    ######### Test Data  ##########

    clients_val_test  = _splitValDataBalanced(df_val_all, 2)




    clients_vals_data = [
        _loadRawData(df) for df in _splitValDataBalanced(clients_val_test[0], num_clients)
    ]

    clients_test_data = [
        _loadRawData(df) for df in _splitValDataBalanced(clients_val_test[1], num_clients)
    ]


    server_data = (_loadRawData(df) for df in _splitValDataBalanced(df_test_all, 2))

    return train_dls, clients_vals_data, clients_test_data, server_data
