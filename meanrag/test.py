import json
import torch
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def data_set():
    # Open and read the JSON file
    with open('contextual_queries.json', 'r') as file:
        # Load the JSON data
        data = json.load(file)

    # Iterate through each record in the list
    for record in data:
        print("Main Query:", record["Main Query"])
        print("Main Query Response:", record["Main Query Response"])
        print("Follow-Up Query:", record["Follow-Up Query"])
        print("Follow Up Query Response:", record["Follow Up Query Response"])
        print("\n" + "-"*50 + "\n")  # Print a separator between records

    print(f'Total Size of dataset is {len(data)}')


    phi2 = "Answe1"
    llma3 = "Aswer2"





def main():
    # Set up the model
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Create a Hugging Face pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    )

    # Create a LangChain HuggingFacePipeline object
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)

    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short paragraph about {topic}:"
    )

    # Create a LangChain
    chain = LLMChain(llm=hf_pipeline, prompt=prompt)

    # Use the chain
    topic = "artificial intelligence"
    result = chain.run(topic)
    print(f"Result for topic '{topic}':")
    print(result)

if __name__ == "__main__":
    main()
    


