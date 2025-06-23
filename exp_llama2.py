import pandas as pd
from utils.dataset import load_datasets
import time
import gc
from transformers import AutoTokenizer, LlamaForCausalLM

import torch
import transformers
import accelerate
from transformers import AutoModel, AutoTokenizer
from llama_index.embeddings import HuggingFaceEmbedding

from tqdm import tqdm
from sentence_transformers import util, SentenceTransformer
from utils.llama2 import getLLAMMA2Model
import sys
# from utils.angle_llama2 import getAngleLlama2


# from utils.llama2 import getLLAMMA2Model


def _reduceData(df, size):
    df = df.sample(frac=1).reset_index(drop=True)
    df1 = df[df["is_duplicate"] == 1].sample(frac=1).reset_index(drop=True)
    df0 = df[df["is_duplicate"] == 0].sample(frac=1).reset_index(drop=True)
    val = pd.concat([df1.head(size), df0.head(size)])
    val = val.sample(frac=1).reset_index(drop=True)
    return val


# def evalLLama2(dname):
#     _, _, _, server_data = load_datasets(dname, 2, 128)
#     val_data, test_data = server_data
#     ts1, ts2, tl = test_data
#     df_test = (
#         pd.DataFrame({"question1": ts1, "question2": ts2, "is_duplicate": tl})
#         .sample(frac=1)
#         .reset_index(drop=True)
#     )
#     df_test = _reduceData(df_test, 100)
#     test_data = (
#         df_test["question1"].tolist(),
#         df_test["question2"].tolist(),
#         df_test["is_duplicate"].tolist(),
#     )

#     vs1, vs2, vl = val_data
#     all_dict = []
#     for t in [i / 100 for i in range(60, 90, 1)]:
#         d = evaluateTransformerModel(
#             None, *test_data, t, hit_miss=True, use_llama2=True
#         )
#         print(d)
#         all_dict.append(d)
#     df = pd.DataFrame(all_dict)
#     print(df)
#     df.to_csv(f"csvs/llama2_{dname}.csv", index=False)

def inferenceTime2LLama2(queries):
    
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").cuda()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    
    start = time.time()
    for prompt in tqdm(queries):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        # Generate
        generate_ids = model.generate(inputs.input_ids, max_length=len(prompt))
        tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = time.time()
    avg_inf_time = (end - start) / len(queries)
    print(f"Avg Inference Time: {(end-start)/len(queries)}")
    return avg_inf_time


def inferenceTimeLLama2(queries):
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        # device_map="auto",
        device=0,
    )
    pipeline.model = pipeline.model.cuda()
    print("pipeline.device: ", pipeline.device)

    start = time.time()
    for q in tqdm(queries):
        _ = pipeline(
            q,
            do_sample=True,
            top_k=1,
            temperature=0.7,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=len(q),  # important point write it in paper
        )

    end = time.time()
    avg_inf_time = (end - start) / len(queries)

    pipeline.device = torch.device("cpu")

    pipeline.model = pipeline.model.cpu()

    print(f"Avg Inference Time: {(end-start)/len(queries)}")
    return avg_inf_time


def computeEmbeddingTime(mname, new_queries, cached_queries):
    def _computeEmbeddingsLLama(q):
        # seq_ids = tokenizer(q, return_tensors="pt")["input_ids"]
        # embedding = embedding_model(seq_ids)["last_hidden_state"].mean(axis=[0, 1]).detach().numpy()
        embedding = embedding_model.get_text_embedding(q)
        
        print(f"llama len of embedding: {len(embedding)}")
        print(f" size of embeddings: {sys.getsizeof(list(embedding))/1024}")
        # print(len(embedding))
        # return embedding
        exit()
        return None

    def _computeEmbeddingSBERTNeT(q):
        embedding = embedding_model.encode(
            q, convert_to_numpy=True, show_progress_bar=False
        )

        assert len(embedding) == 768

        print(f"len of embedding: {len(embedding)}")
        print(f"size of embeddings: {sys.getsizeof(list(embedding))/1024}")
        exit()
        # return embedding
        return None

    _computeEmbeddings = None
    embedding_model = None
    if mname == "llama2":
        print("------------------------------------ Alert USE LLAMA2 on A100")

        embedding_model = getLLAMMA2Model()
        # embedding_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        _computeEmbeddings = _computeEmbeddingsLLama
    elif mname == "albert":
        embedding_model = SentenceTransformer("paraphrase-albert-small-v2")
        embedding_model = embedding_model.cuda()
        _computeEmbeddings = _computeEmbeddingSBERTNeT
    elif mname == "mpnet":
        embedding_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
        embedding_model = embedding_model.cuda()
        _computeEmbeddings = _computeEmbeddingSBERTNeT

    start = time.time()
    cached_embeddings = [_computeEmbeddings(q) for q in tqdm(cached_queries)]
    end = time.time()

    avg_embedding_computation_time = (end - start) / len(cached_queries)    
    return avg_embedding_computation_time


def main():
    _, _, _, server_data = load_datasets("dgptcache", 2, 128)
    val_data, test_data = server_data
    new_queries2, cached_queries, labels = test_data

    cached_queries = new_queries2[:1000]
    new_queries = cached_queries
    # cached_queries = cached_queries + new_queries2
    # avg_inf_time = inferenceTimeLLama2(cached_queries)
    # print(f"Avg. Inference Time: {avg_inf_time}")

    # avg_llama2_emb_time = computeEmbeddingTime(
    #     "llama2", new_queries=new_queries, cached_queries=cached_queries
    # )
    
    # avg_albert_emb_time = computeEmbeddingTime(
    #     "albert", new_queries=new_queries, cached_queries=cached_queries
    # )
    
    avg_mpnet_emb_time = computeEmbeddingTime(
        "mpnet", new_queries=new_queries, cached_queries=cached_queries
    )

    

    df = pd.DataFrame(
        {
            # "Inference Time (s)": [avg_inf_time],
            "LLama2 Embedding Computation Time (s)": [avg_llama2_emb_time],
            "Albert Embedding Computation Time (s)": [avg_albert_emb_time],
            "MPNet Embedding Computation Time (s)": [avg_mpnet_emb_time],
        }
    )
    print(df)

    df.to_csv("csvs/embeddings_computatons_llama2_infertime.csv", index=False)


if __name__ == "__main__":
    main()
