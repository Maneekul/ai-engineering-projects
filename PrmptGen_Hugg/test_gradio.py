import os
from langchain_huggingface import HuggingFaceEndpoint
import pandas as pd
import numpy as np
import gradio as gr
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your Hugging Face API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

def setup_llm_chain(model_repo, temperature, top_p, max_new_tokens, repetition_penalty, custom_prompt):
    llm = HuggingFaceHub(
        repo_id=model_repo,
        model_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty
        }
    )

    prompt = PromptTemplate(input_variables=["input"], template=custom_prompt)
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain

def generate_prompt(user_input, custom_prompt, max_tokens):
    # Model parameters 
    model_repo = "meta-llama/Llama-3.1-8B"   #meta-llama/Llama-3.2-1B  mistralai/Mixtral-8x7B-Instruct-v0.1
    temperature = 0.3
    top_p = 0.95
    repetition_penalty = 1.1

    # Setup LLM chain
    prompt_generator_chain = setup_llm_chain(model_repo, temperature, top_p, max_tokens, repetition_penalty, custom_prompt)

    # Generate prompt
    generated_prompt = prompt_generator_chain.run(user_input)
    
    return generated_prompt

# Default prompt template
default_prompt = """
YOU ARE AN EXPERT IN TEXT LAYOUT AND DESIGN. YOUR TASK IS TO ENLARGE THE FONT SIZE OF A GIVEN TEXT WHILE MAINTAINING THE LINE BREAKS.

  ###INSTRUCTIONS###
  - Analyze the text and determine the current font size.
  - Compute the desired font size based on the given ratio.
  - Adjust the font size while maintaining line breaks.

  ###Chain of Thoughts###
  1. Read and understand the text and the desired ratio for enlargement.
  2. Calculate the new font size based on the provided ratio.
  3. Modify the font size accordingly without disturbing the line breaks.

  ###What Not To Do###
  - Change font sizes randomly.
  - Enlarge font sizes without calculating the appropriate size.
  - Ignore line breaks during the enlargement process.
  - Forget to check and adjust the final layout.

{input}

"""

iface = gr.Interface(
    fn=generate_prompt,
    inputs=[
        gr.Textbox(lines=3, label="Enter your question"),
        gr.Textbox(lines=10, label="Custom Prompt Template (use {input} for user input placement):", value=default_prompt),
        gr.Slider(100, 4096, value=1000, label="Max Tokens")
    ],
    outputs=gr.Textbox(lines=10, label="Generated Advice:"),
    title="AI Generator Prompts",
    description="Get expert advice on opening and managing a Thai food business."
)

iface.launch()