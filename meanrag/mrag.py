# import torch
import json
from typing import Dict, List
# from langchain.llms import HuggingFacePipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from rouge_score import rouge_scorer
from diskcache import Index
import ollama
import time
import pandas as pd
import ast

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import subprocess
import concurrent.futures

import hashlib

import logging

def hash_string(input_string, algorithm='sha256'):
    # Create a hash object
    hash_object = hashlib.new(algorithm)
    
    # Convert the input string to bytes and update the hash object
    hash_object.update(input_string.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    hashed_string = hash_object.hexdigest()
    # print(type(hashed_string))
    return hashed_string

def load_dataset(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

# def setup_model(model_id: str):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device) 



#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=256,  # Generate up to 100 new tokens
#         # do_sample=True,
#         # temperature=0.7,
#         # top_k=50,
#         # top_p=0.95,
#         # num_return_sequences=1,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device=device
#     )
    
#     return HuggingFacePipeline(pipeline=pipe)

# def generate_response(llm, main_query: str, main_response: str, follow_up_query: str) -> str:
#     # prompt = PromptTemplate(
#     #     input_variables=["main_query", "main_response", "follow_up_query"],
#     #     template="Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context:\nMain Query: {main_query}\nMain Query Response: {main_response}\n\nNow, answer the following follow-up query:\nFollow-up Query: {follow_up_query}\n\n. Answer:"
#     # )

#     # template = """Use the following pieces of context to answer the question at the end. 
#     # If you don't know the answer, just say that you don't know, don't try to make up an answer. 
#     # Use at most 512 words maximum and keep the answer as concise as possible. 
#     # {context}
#     # Question: {question}
#     # Answer:"""

#     template = """Answer the question based only on the following context:
#     {context}. If you cannot answer based on the context, please say ```Unable to generate response```.
#     Question: {question}. Just give the plain answer do not say anything else.
#     """

#     prompt =  PromptTemplate(
#         input_variables=["context", "question"],
#         template=template,
#     )



#     chain = LLMChain(llm=llm, prompt=prompt)
#     result = chain.run(context=f"{main_query}-Response:{main_response}", question=follow_up_query)
#     return result.strip()
    
# def evaluate_response(generated: str, ground_truth: str) -> Dict[str, float]:
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = scorer.score(ground_truth, generated)
#     return {key: value.fmeasure for key, value in scores.items()}

# def main3():
#     dataset = load_dataset('context2.json')  # Assuming the dataset is stored in a JSON file
    
#     models = ["meta-llama/Meta-Llama-3-8B" ]  # Add more models as needed
    
#     results = {}
    
#     for model_id in models:
#         print(f"Evaluating model: {model_id}")
#         llm = setup_model(model_id)
        
#         model_results = []
        
#         for item in dataset:
#             generated_response = generate_response(
#                 llm, 
#                 item['Main Query'], 
#                 item['Main Query Response'], 
#                 item['Follow-Up Query']
#             )

#             print(f"Generated Response: {generated_response}")
            
#             scores = evaluate_response(generated_response, item['Follow Up Query Response'])
            
#             model_results.append({
#                 'Main Query': item['Main Query'],
#                 'Follow-Up Query': item['Follow-Up Query'],
#                 'Ground Truth': item['Follow Up Query Response'],
#                 'Generated Response': generated_response,
#                 'Scores': scores
#             })
        
#         results[model_id] = model_results
    
#     # # Print or save results
#     # for model_id, model_results in results.items():
#     #     print(f"\nResults for {model_id}:")
#     #     for result in model_results:
#     #         print(f"Main Query: {result['Main Query']}")
#     #         print(f"Follow-Up Query: {result['Follow-Up Query']}")
#     #         print(f"Ground Truth: {result['Ground Truth']}")
#     #         print(f"Generated Response: {result['Generated Response']}")
#     #         print(f"Scores: {result['Scores']}")
#     #         print("---")








def _convert_to_python_dict(score_string):
    # Removing the brackets and splitting the string
    print(score_string)
    cleaned_str = score_string.strip("[]")
    items = cleaned_str.split(", ")

    # Converting the split items into a dictionary
    result_dict = {}
    for item in items:
        key, value = item.split(": ")
        result_dict[key] = int(value)

    # Displaying the dictionary
    # print(result_dict)
    return result_dict


def _helperdownload_model(model_name):
    command = f"ollama pull {model_name}"
    print(f"Downloading {model_name}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded {model_name}")
        return f"{model_name}: Success"
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {model_name}: {e}")
        return f"{model_name}: Failed - {e}"

def download_model(all_models):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_helperdownload_model, model) for model in all_models.keys()]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    print("\nDownload Results:")
    print(json.dumps(results, indent=2))


def generate_response(template, mname, check_cache):
    cache  = Index(f'cache_responses/{mname}')
    
    key = hash_string(template)
    if check_cache==1 and key in cache:
        return cache[key] 
    
    start_time = time.time()
    response = ollama.chat(
        model=mname,
        messages=[{'role': 'user', 'content': template}]

    )
    end_time = time.time()
    # print(f'-----> llm response: {response}')
    tokens_per_second = response['eval_count']/response['eval_duration'] * 10e9
    # print(f'-Total duration : {response["eval_duration"]+ response["prompt_eval_duration"]+ response["load_duration"]}')


    res = {f'Response': response['message']['content'], f'Response-Time':response['eval_duration'], 'Tokens/seconds': tokens_per_second, 'template': template, 'response-dict': response}
    cache[key] = res
    return res


def generate_response_context(mname,cont_dict, check_cache):
    
    main_query =  cont_dict['Main Query'], 
    main_response= cont_dict['Main Query Response'], 
    follow_up_query = cont_dict['Follow-Up Query']

    context = f"{main_query}-Response:{main_response}"
    question = follow_up_query
    template = f"Answer the question based only on the following context: {context}. If you cannot answer based on the context, please say ```Response cannot be returned from cache.```. Question: {question}. Do not write anything else."

    return generate_response(template, mname, check_cache)
    

def generate_response_withoutcontext(mname,cont_dict, check_cache):
    cache  = Index(f'cache_responses/{mname}')
    
    main_query =  cont_dict['Main Query'], 
    main_response= cont_dict['Main Query Response'], 
    follow_up_query = cont_dict['Follow-Up Query']

    context = f"{main_query}-Response:{main_response}"
    question = follow_up_query
    template = f"Answer the question. Question: {question}. Do not write anything else."
    return generate_response(template, mname, check_cache)


def llm_judge(llm_judge, r1, r2, check_cache):
    # prompt = f"Use `Ground Truth Response` as a refrence and evaluate the `Generated Response` based on correctness, relevance, coherence, and accuracy. Provide a score from 1 to 10 for each criterion. Ground Truth Response: {r1}\n Generated Response: {r2}. \n Your output should exactly like this [Correctness: 8, Relevance: 7, Coherence: 6, Accuracy: 9]. \n Do not give any explaination."

    cache = Index(f'cache_responses/{llm_judge}')
    prompt = f"Use `Ground Truth Response` as a refrence and evaluate the `Generated Response` based on correctness and relevance. Ground Truth Response: {r1}\n Generated Response: {r2} Give your answer as a float on a scale of 0 to 5, where 0 means that the `Generated Response` is not helpful at all and 5 means that it is relevant and fully helpful. \n Do not give any explaination. \n\n Just output the the score as follows: \n Score: x where x is a float between 0 and 10. Do not write anything else."
    key = hash_string(prompt)

    if check_cache==1 and key in cache:
        return cache[key]

    response = ollama.chat(
        model=llm_judge,
        messages=[{'role': 'user', 'content': prompt}]
    )

    score =  response['message']['content']
    score =  float(score.split(':')[-1])
    cache[key] = score
    return score


    

def main2_ollama(cfg,all_models, queries):
    all_rows = []
    for row_dict in tqdm(queries):
        for model_name, msize in all_models.items():
            r = generate_response_context(model_name,row_dict, cfg.check_cache)
            row_dict[f'Generated-Response Context ({model_name})'] = r['Response']
            row_dict[f'Response-Time Context ({model_name})'] = r['Response-Time']
            row_dict[f'Tokens/seconds Context ({model_name})'] = r['Tokens/seconds']
            score = llm_judge(cfg.judge_llm ,row_dict['Follow Up Query Response'], r['Response'], cfg.check_cache)
            row_dict[f'Scores Context ({model_name})'] = score
            row_dict[f'Model {model_name} Size'] = msize
            r2 = generate_response_withoutcontext(model_name,row_dict, cfg.check_cache)
            row_dict[f'Generated-Response Without Context ({model_name})'] = r2['Response']
            row_dict[f'Response-Time Without Context ({model_name})'] = r2['Response-Time']
            row_dict[f'Tokens/seconds Without Context ({model_name})'] = r2['Tokens/seconds']
            score2 = llm_judge(cfg.judge_llm ,row_dict['Follow Up Query Response'], r2['Response'], cfg.check_cache)
            row_dict[f'Scores Without Context ({model_name})'] = score2
            # row_dict[f'Model {model_name} Size'] = msize

                
        all_rows.append(row_dict)
    df = pd.DataFrame(all_rows)
    df.to_csv(cfg.results_csv_name, index=False)


        
@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg):

    all_models=  {'llama3': 8.03, 'gemma:2b':2.51, 'gemma:7b':8.54, 'qwen2:0.5b':494/1000, 'qwen2:1.5b':1.54, 'qwen2:7b':7.62, 'tinyllama':1.10, 'phi3': 3.83, 'phi3:14b': 14}


    if cfg.download_models==1:
        # print(all_models)
        download_model(all_models)
    all_queries = [] 
    for json_file in cfg.json_files:
        all_queries.extend(load_dataset(json_file))
    
    df = pd.DataFrame(all_queries)

    logging.info('Before removing duplicates: %s', df.shape)


    df.drop_duplicates(subset=['u0'], inplace=True)
    logging.info('After removing duplicates: %s', df.shape)

    queries = df.to_dict(orient='records')
    # main2_ollama(cfg,all_models, queries)



    





if __name__ == "__main__":
    # main()
    main()