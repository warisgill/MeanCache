from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token
embed_model = HuggingFaceEmbedding(model=model, model_name='meta-llama/Llama-2-7b-chat-hf', tokenizer=tokenizer, tokenizer_name='meta-llama/Llama-2-7b-chat-hf')

print("Embedding model loaded device: ", embed_model._device)

# embeddings1 = embed_model.get_text_embedding("Hello World!")
# emeddings2 = embed_model.get_text_embedding("theire is a good nighe .. hello how are you, Hello World 2!")



def getLLAMMA2Model ():
    return embed_model


# from transformers import AutoModel, AutoTokenizer
# model = AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf').cpu()
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
# tokenizer.pad_token = tokenizer.eos_token
# embed_model = HuggingFaceEmbedding(model=model, model_name='meta-llama/Llama-2-7b-chat-hf', tokenizer=tokenizer, tokenizer_name='meta-llama/Llama-2-7b-chat-hf')

# embeddings1 = embed_model.get_text_embedding("Hello World!")
# emeddings2 = embed_model.get_text_embedding("theire is a good nighe .. hello how are you, Hello World 2!")

# def getLLAMMA2Model ():
#     return embed_model

# text = "Hello World!"

# model = "meta-llama/Llama-2-7b-chat-hf"
# full_model = AutoModel.from_pretrained(model)
# tokenizer = AutoTokenizer.from_pretrained(model)
# seq_ids = tokenizer(text, return_tensors='pt')["input_ids"]
# embedding = full_model(seq_ids)["last_hidden_state"].mean(axis=[0,1]).detach().numpy()

# print(f"len of embedding: {len(embedding)}")

# model = "meta-llama/Llama-2-7b-chat-hf"
# full_model = AutoModel.from_pretrained(model).to("cuda:0")
# tokenizer = AutoTokenizer.from_pretrained(model)

# def llama2Embedding (text):
#     seq_ids = tokenizer(text, return_tensors='pt').to("cuda:0")["input_ids"]
#     embedding = full_model(seq_ids)["last_hidden_state"].mean(axis=[0,1]).cpu().detach().numpy()
#     return embedding

# e1 =  llama2Embedding("Hello World!")
# e2 =  llama2Embedding("theire is a good nighe .. hello how are you, Hello World 2!")

# print(len(e1))
# print(len(e2))

