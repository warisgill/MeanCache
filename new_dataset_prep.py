import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM

# https://www.philschmid.de/llama-2






class NewCacheDataset:
    def __init__(self):
        self.labels = []
        self.orignal_queries = []
        self.duplicate_queries = []
        self.source_name = None
        self.words_in_queries = [] 
        dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
        self.llama2 =  LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").cuda()
        self.llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        # print(dataset)

        for it in dataset['train']:
            # print(it['conversations'])
            for conv_msg in it['conversations']:
                if conv_msg['from'].find('human') != -1:
                    q =  conv_msg['value'].strip()
                    self.words_in_queries.append(len(q.split()))
                    self.orignal_queries.append(q)
                    break
                    
            # _ = input("Press Enter to continue...")
            # print(">>\n\n")
            # break
        print(f"Total Number of Orignal Queries: {len(self.orignal_queries)}")


    def paraphrase(self):
        # Define the prompt template

        # Replace placeholders with actual content
        system_prompt = "Only paraphrase the text. Ouput format is <paraphrase>:{your response}"
        # user_message = "I would like to know the current weather."
        

        # prompts = [prompt_template]

        for query in self.orignal_queries:
            prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>```paraphrase { query }```[/INST]"
            inputs = self.llama2_tokenizer(prompt, return_tensors="pt").to("cuda:0")
            # Generate
            generate_ids = self.llama2.generate(inputs.input_ids, max_length=len(prompt))
            resp =  self.llama2_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            assert resp.find("<paraphrase>:") != -1
            paraphrase_q =  resp.split("<paraphrase>:")[-1]

            print(f'orignal query ====> {query}') 
            print(f"phraased =====> {paraphrase_q}")
            _ = input(">> nter to continue \n\n")
        
        return texts


    



    def convertToCSv(self):
        d = {'question1':self.orignal_queries , 'question2':self.duplicate_queries, 'is_duplicate':self.labels, 'source':self.source_name}
        df = pd.DataFrame(data=d)
        df.to_csv(self.csv_path, index=False)


        
        
d = NewCacheDataset()
d.paraphrase()


