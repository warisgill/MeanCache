from angle_emb import AnglE, Prompts



def getAngleLlama2():
    angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2')
    print('All predefined prompts:', Prompts.list_prompts())
    angle.set_prompt(prompt=None)
    print('prompt:', angle.prompt)
    vec = angle.encode('Hello how are you', to_numpy=True)
    print(len(vec[0]))
    # print(vec)
    vecs = angle.encode(['hello world1', 'hello world2'], to_numpy=True)
    # print(vecs)
    print(len(vecs))
    return angle