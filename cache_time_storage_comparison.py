import pandas as pd
import time
import torch
import transformers
from datetime import datetime

# import accelerate
from utils.dataset import load_datasets
from transformers import AutoModel, AutoTokenizer


from tqdm import tqdm

from utils.eutil import (
    TupleToDFDataset,
    dfToTupleDataset,
    getEvalMetrics,
)
from utils.caches import GPTCache, FedGPTCache, FedGPTCacheCompression


class LLama2Service:
    def __init__(self, cache, prev_times=[]):
        self.cache = cache
        self.llama2_pipeline = None
        self.tokenizer = None
        self._setLLama2Pipeline()
        self.predicted_labels = []
        self.prev_times = prev_times

    def _setLLama2Pipeline(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        # self.llama2_pipeline = transformers.pipeline(
        #     "text-generation",
        #     model="meta-llama/Llama-2-7b-chat-hf",
        #     torch_dtype=torch.float16,
        #     # device_map="auto",
        #     device="cuda:0",
        # )
        return

    def _generateResponse(self, query):
        # print("Generating Response")
        # start = time.time()
        # r = self.llama2_pipeline(
        #     query,
        #     do_sample=True,
        #     top_k=1,
        #     temperature=0.7,
        #     num_return_sequences=1,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     max_length=50,  # important point write it in paper
        # )
        # endtime = time.time() - start
        # print(f"response: {r}")
        return 0, 0

    def _getResponseFromCache(self, query, q_index, q_label):
        cache_r, cache_time = self.cache.getResponse(query)
        llama_response = "Cache Response"
        llama_time = 0

        if cache_r != -1 and q_label == 1:
            if cache_r == q_index:
                self.predicted_labels.append(1)
                return cache_r, cache_time
            else:
                self.predicted_labels.append(0)
                # llama_response, llama_time = self._generateResponse(query)
                llama_time = self.prev_times[q_index]
                llama_response = "LLama2 Response"
                # return llama_response,  2 * (llama_time + cache_time)
                return llama_response, llama_time + 2 * cache_time

        elif cache_r == -1 and q_label == 0:
            self.predicted_labels.append(0)
            llama_time = self.prev_times[q_index]
            llama_response = "LLama2 Response"
            # llama_response, llama_time = self._generateResponse(query)
            return llama_response, llama_time + cache_time

        elif cache_r == -1 and q_label == 1:
            self.predicted_labels.append(0)
            # llama_response, llama_time = self._generateResponse(query)
            llama_time = self.prev_times[q_index]
            llama_response = "LLama2 Response"
            return llama_response, llama_time + cache_time

        elif cache_r != -1 and q_label == 0:
            self.predicted_labels.append(1)
            # llama_response, llama_time = self._generateResponse(query)
            llama_time = self.prev_times[q_index]
            llama_response = "LLama2 Response"
            # return llama_response, 2 * (llama_time + cache_time)
            return llama_response, llama_time + 2 * cache_time

        else:
            raise Exception("Invalid query label")

    def sendQuery(self, query, q_index, q_label):
        t = -1
        r = None
        if self.cache is None:
            r, t = self._generateResponse(query)
        else:
            r, t = self._getResponseFromCache(query, q_index, q_label)
        return r, t

    def getPredictedLabels(self):
        return self.predicted_labels


def main():
    def evalQueries(llama2_service):
        response_time = []
        for i in tqdm(range(len(new_queries2))):
            r, t = llama2_service.sendQuery(new_queries2[i], i, labels[i])
            response_time.append(t)
        pred_labels = llama2_service.getPredictedLabels()
        return response_time, pred_labels

    _, _, _, server_data = load_datasets("dgptcache", 2, 128)
    val_data, test_data = server_data
    df = TupleToDFDataset(*test_data)

    percent_sim = 0.3
    all_dict = []

    # total_eval_size = 1000
    currentTime_GMT = f"{datetime.now().timestamp()}"
    currentTime_GMT = currentTime_GMT.split(".")[0]
    for total_eval_size in range(500, 4001, 500):
        # sample similalr queries
        df1 = df[df["is_duplicate"] == 1].head(int(percent_sim * total_eval_size))
        df0 = df[df["is_duplicate"] == 0].head(
            total_eval_size - int(percent_sim * total_eval_size)
        )

        df_test = pd.concat([df0, df1]).reset_index(drop=True)

        print(f"Count Values of df_test: {df_test['is_duplicate'].value_counts()}")

        new_queries2, cached_queries, labels = dfToTupleDataset(df_test)

        llama2_res_times = [0] * len(labels)
        # -------------------- Cachesh eval  --------------------------
        cache = GPTCache(cached_queries)
        _, pred_labels = evalQueries(
            llama2_service=LLama2Service(cache, llama2_res_times)
        )
        eval_dict_cache = getEvalMetrics(labels, pred_labels)
        # eval_dict_cache["Avg. Search Time (s)"] = cache.cache.avgQueryTime()
        eval_dict_cache.update(cache.cache.avgQueryTime())
        eval_dict_cache["Storage Size (KBs)"] = cache.cache.getStorageSize()["KBs"]
        eval_dict_cache["Size"] = len(labels)
        eval_dict_cache["Config"] = "GPTCache"
        all_dict.append(eval_dict_cache)

        key = "15th:tname-multi-qa-mpnet-base-cos-v1-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"
        cache = FedGPTCache(cached_queries, key=key, optimal_threshold=0.83)
        _, pred_labels = evalQueries(
            llama2_service=LLama2Service(cache, llama2_res_times)
        )
        eval_dict_cache = getEvalMetrics(labels, pred_labels)
        # eval_dict_cache["Avg. Search Time (s)"] = cache.cache.avgQueryTime()
        eval_dict_cache.update(cache.cache.avgQueryTime())
        eval_dict_cache["Storage Size (KBs)"] = cache.cache.getStorageSize()["KBs"]
        eval_dict_cache["Size"] = len(labels)
        eval_dict_cache["Config"] = "FedGPTCache (mpnet)"
        all_dict.append(eval_dict_cache)

        albert_key = "15th:tname-paraphrase-albert-small-v2-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"
        key = albert_key
        cache = FedGPTCache(cached_queries, key=key, optimal_threshold=0.78)
        _, pred_labels = evalQueries(
            llama2_service=LLama2Service(cache, llama2_res_times)
        )
        eval_dict_cache = getEvalMetrics(labels, pred_labels)
        # eval_dict_cache["Avg. Search Time (s)"] = cache.cache.avgQueryTime()
        eval_dict_cache.update(cache.cache.avgQueryTime())
        eval_dict_cache["Storage Size (KBs)"] = cache.cache.getStorageSize()["KBs"]
        eval_dict_cache["Size"] = len(labels)
        eval_dict_cache["Config"] = "FedGPTCache (albert)"
        all_dict.append(eval_dict_cache)

        key = "15th:tname-multi-qa-mpnet-base-cos-v1-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"
        cache = FedGPTCacheCompression(
            cached_queries,
            compression_queries=val_data[0] + val_data[1],
            compressing_dim=128,
            key=key,
            optimal_threshold=0.83,
        )
        _, pred_labels = evalQueries(
            llama2_service=LLama2Service(cache, llama2_res_times)
        )
        eval_dict_cache = getEvalMetrics(labels, pred_labels)
        eval_dict_cache.update(cache.cache.avgQueryTime())
        eval_dict_cache["Storage Size (KBs)"] = cache.cache.getStorageSize()["KBs"]
        eval_dict_cache["Size"] = len(labels)
        eval_dict_cache["Config"] = "FedGPTCache-Compressed (mpnet)"
        all_dict.append(eval_dict_cache)

        albert_key = "15th:tname-paraphrase-albert-small-v2-dname-dgptcache-clients_per_round-4-num_clients-20-batch_size-128-device-cuda-client_epochs-6-num_rounds-50-loss_type-both-mnr-contrastive-"
        key = albert_key
        cache = FedGPTCacheCompression(
            cached_queries,
            compression_queries=val_data[0] + val_data[1],
            compressing_dim=128,
            key=key,
            optimal_threshold=0.78,
        )
        _, pred_labels = evalQueries(
            llama2_service=LLama2Service(cache, llama2_res_times)
        )
        eval_dict_cache = getEvalMetrics(labels, pred_labels)
        # eval_dict_cache["Avg. Search Time (s)"] = cache.cache.avgQueryTime()
        eval_dict_cache.update(cache.cache.avgQueryTime())
        eval_dict_cache["Storage Size (KBs)"] = cache.cache.getStorageSize()["KBs"]
        eval_dict_cache["Size"] = len(labels)
        eval_dict_cache["Config"] = "FedGPTCache-Compressed (albert)"
        all_dict.append(eval_dict_cache)

    df = pd.DataFrame(all_dict)

    df.to_csv(f"csvs/time_storage_f1_comparison_{currentTime_GMT}.csv")


if __name__ == "__main__":
    main()
