from diskcache import Index
import time
from sentence_transformers import util, SentenceTransformer, models
from sklearn.decomposition import PCA
import torch
import numpy as np
import sys
import logging
from transformers import AutoModel, AutoTokenizer
import time
import torch
import transformers


global_model_cache = Index(".storage/cache/global_models")


class EmbeddingContextualCache:
    def __init__(self, model, cached_queries, cossim_t, top_k):
        self.cossim_t = cossim_t
        self.model = model
        self.model.eval()
        self.query_times = []
        self.query_and_inf_time = []
        self.mcache = []
        self._populate_cache(cached_queries)
        self.embeds = [x['embedding'] for x in self.mcache]
        self.top_k = top_k

    def _populate_cache(self, queries):
        i = 0
        for _,q in enumerate(queries):
            cached_entry_parent = {'index':i,  'query': q['u0'], 'embedding': self.model.encode(q['u0'], convert_to_numpy=False, show_progress_bar=False),'parent_index':-1, 'metadata': q}
            self.mcache.append(cached_entry_parent)
            cached_entry_child = {'index':i+1,   'query': q['u1'], 'embedding': self.model.encode(q['u1'], convert_to_numpy=False, show_progress_bar=False),'parent_index':i, 'metadata': q}
            i += 2
            self.mcache.append(cached_entry_child)
        
        logging.info(f"Cache populated with {len(queries)} queries")
        logging.info(f"Cache populated with {len(self.mcache)} entries")


    

    
    def get_avg_query_time(self):
        return {"Avg. Search Time (s)": sum(self.query_times)/len(self.query_times), 
                "Avg. Infer+Search Time(s)": sum(self.query_and_inf_time)/len(self.query_and_inf_time)
                }    
    def get_storage_size(self):
        # Calculate the size of l1 in bytes
        size_in_bytes = sys.getsizeof(self.embeds)

        # Convert the size to kilobytes
        size_in_kilobytes = size_in_bytes / 1024
        return {"KBs": size_in_kilobytes}
    
    def _is_embedding_cached(self, query_embedding):
        start_time = time.time()
        top_k_hits =  util.semantic_search(query_embedding, self.embeds, top_k= self.top_k)
        final_hits = []
        for top_k in top_k_hits:
            for match in top_k:
                if match['score'] >= self.cossim_t:
                    final_hits.append(match)
        self.query_times.append(time.time()-start_time)
        
        if len(final_hits) > 0:
            return [self.mcache[d['corpus_id']] for d in final_hits]
        return []
    
    def get_cache_hits(self, query):
        start_time = time.time()
        query_emb = self.model.encode(query, convert_to_numpy=False, show_progress_bar=False)
        hits = self._is_embedding_cached(query_emb)
        end_time = time.time()
        self.query_and_inf_time.append(end_time-start_time)
        return hits, end_time - start_time





class EmbeddingCache:
    def __init__(self, model, cached_queries, cossim_t):
        self.cossim_t = cossim_t
        self.model = model
        self.model.eval()
        self.embeddings = self.model.encode(cached_queries, convert_to_numpy=True)
        print(f"len of single embedding: {len(self.embeddings[0])}")
        self.query_times = []
        self.query_and_inf_time = []

    def isQueryCached(self, query):
        start_time = time.time()
        query_emb = self.model.encode(query)
        r_i = self._isEmbeddingCached(query_emb)
        end_time = time.time()
        self.query_and_inf_time.append(end_time-start_time)
        return r_i, end_time - start_time

    def _isEmbeddingCached(self, query_embedding):
        start_time = time.time()
        all_scores_cache =  util.cos_sim(query_embedding, self.embeddings)
        for i in range(len(self.embeddings)):
            # cos_score = util.pytorch_cos_sim(query_embedding, self.embeddings[i])
            cos_score = all_scores_cache[0][i]
            if cos_score > self.cossim_t:
                return i
        self.query_times.append(time.time()-start_time)
        return -1
    def avgQueryTime(self):
        return {"Avg. Search Time (s)": sum(self.query_times)/len(self.query_times), 
                "Avg. Infer+Search Time(s)": sum(self.query_and_inf_time)/len(self.query_and_inf_time)
                }    
    def getStorageSize(self):
        # Calculate the size of l1 in bytes
        size_in_bytes = sys.getsizeof(self.embeddings)

        # Convert the size to kilobytes
        size_in_kilobytes = size_in_bytes / 1024
        return {"KBs": size_in_kilobytes}



class MCacheCTX:
    def __init__(self, mname, cached_queries, cossim_t, top_k):
        # self.cossim_t = cossim_t
        self.model = SentenceTransformer(mname).cpu() 
        self.context_cache = EmbeddingContextualCache(self.model, cached_queries, cossim_t, top_k=top_k)
    
    def get_response(self, query, lineage_id):
        potential_cache_hits, time_taken =  self.context_cache.get_cache_hits(query)

        # predicting the true hit
        for hit in potential_cache_hits:
            p_i = hit['parent_index']
            if p_i == -1 or p_i == lineage_id:   
                return {'predicted_index': hit['index'], 'metadata': hit['metadata'], 'time': time_taken}
             
        # if len(potential_cache_hits) > 0 and lineage_id != -2:
        #     hit = potential_cache_hits[0]
        #     return {'predicted_index': hit['index'], 'metadata': hit['metadata'], 'time': time_taken}
        return {'predicted_index': -1, 'metadata': None, 'time': time_taken}


class GPTCacheCTX(MCacheCTX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_response(self, query, lineage_id=None):
        potential_cache_hits, time_taken =  self.context_cache.get_cache_hits(query)
        if len(potential_cache_hits) > 0:
            hit = potential_cache_hits[0]
            return {'predicted_index': hit['index'], 'metadata': hit['metadata'], 'time': time_taken}
        return {'predicted_index': -1, 'metadata': None, 'time': time_taken}





class GPTCache:
    def __init__(self, cached_queries) -> None:
        self.model = SentenceTransformer("paraphrase-albert-small-v2").cpu()
        self.cache = EmbeddingCache(self.model, cached_queries, 0.7)

    def getResponse(self, query):
        return self.cache.isQueryCached(query)


class FedGPTCache:
    def __init__(self, cached_queries, key, optimal_threshold):
        self.model = global_model_cache[key][0]
        self.model.eval()
        self.model = self.model.cpu()
        self.cache = EmbeddingCache(self.model, cached_queries, optimal_threshold)

    def getResponse(self, query):
        return self.cache.isQueryCached(query)

def compressEmbeddings(model, val_qs, new_dimension):
    model = model.cpu()
    model.eval()
    train_embeddings = model.encode(val_qs, convert_to_numpy=True, show_progress_bar=False)
    pca = PCA(n_components=new_dimension)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    dense = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=new_dimension,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module("dense", dense)

    model.eval()
    model = model.cpu()
    return model


class FedGPTCacheCompression:
    def __init__(self, cached_queries, compression_queries, compressing_dim, key, optimal_threshold):
        model = global_model_cache[key][0]
        model =  model.eval().cpu()
        print("len of old model embed size", len(model.encode(["hello"])[0]))
        self.model = compressEmbeddings(model, compression_queries, compressing_dim)
        print("len of new model embed size", len(self.model.encode(["hello"])[0]))
        self.cache = EmbeddingCache(self.model, cached_queries, optimal_threshold)

    def getResponse(self, query):
        return self.cache.isQueryCached(query)
    


class LLama2Service:
    def __init__(self, cache, prev_times=[]):
        self.cache = cache
        self.llama2_pipeline = None
        self.tokenizer = None
        # self._setLLama2Pipeline()
        self.predicted_labels = []
        self.prev_times = prev_times

    def _setLLama2Pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.llama2_pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            # device_map="auto",
            device="cuda:0",
        )

    def _generateResponse(self, query):
        # print("Generating Response")
        start = time.time()
        # r = self.llama2_pipeline(
        #     query,
        #     do_sample=True,
        #     top_k=1,
        #     temperature=0.7,
        #     num_return_sequences=1,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     max_length=50,  # important point write it in paper
        # )
        r = "This is a response. Uncomment the pipeline code to get real response"
        endtime = time.time() - start
        # print(f"response: {r}")
        return r, endtime
    
    def _predict_response_from_cache(self, args_dict):
        query = args_dict['query']
        true_label = args_dict['true_label']
        true_index = args_dict['true_index']
        lineage_id = args_dict['parent_id']    
        res = self.cache.get_response(query, lineage_id=lineage_id)
        # logging.warning('fix query[index]')
                
        llama_response = "Cache Response"
        llama_time = 0

        if res['predicted_index'] != -1 and true_label == 1:
            if res['predicted_index'] == true_index:
                self.predicted_labels.append(1)
                return res['predicted_index'], res['time']
            else:
                self.predicted_labels.append(0)
                # llama_response, llama_time = self._generateResponse(query)
                llama_time = -1 #self.prev_times[true_index]
                llama_response = "LLama2 Response"
                # return llama_response,  2 * (llama_time + res['time'])
                return llama_response,  llama_time +  2 * res['time']

        elif res['predicted_index'] == -1 and true_label == 0:
            self.predicted_labels.append(0)
            llama_time = -1 #self.prev_times[true_index]
            llama_response = "LLama2 Response"
            # llama_response, llama_time = self._generateResponse(query)
            return llama_response, llama_time + res['time']

        elif res['predicted_index'] == -1 and true_label == 1:
            self.predicted_labels.append(0)
            # llama_response, llama_time = self._generateResponse(query)
            llama_time = -1 #self.prev_times[true_index]
            llama_response = "LLama2 Response"
            return llama_response, llama_time + res['time']

        elif res['predicted_index'] != -1 and true_label == 0:
            self.predicted_labels.append(1)
            # llama_response, llama_time = self._generateResponse(query)
            llama_time = -1 #self.prev_times[true_index]
            llama_response = "LLama2 Response"
            # return llama_response, 2 * (llama_time + res['time'])
            return llama_response, llama_time +  2 *res['time']

        else:
            raise Exception("Invalid query label")

    

    def getPredictedLabels(self):
        return self.predicted_labels
    
    def send_query(self, args):
        t = -1
        r = None
        if self.cache is None:
            r, t = self._generateResponse(args['query'])
        else:
            r, t = self._predict_response_from_cache(args)
        return r, t
