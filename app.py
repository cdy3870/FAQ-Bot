import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import gradio as gr


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

PEFT_MODEL = "cdy3870/Falcon-Fetch-Bot"

config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True, cache_dir='E:\code_projects\cache'
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 150
generation_config.temperature = 0.6
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def main():

    with gr.Blocks() as demo:

        def update_temp(temp):
            generation_config.temperature = temp

        def update_tokens(tokens):
            generation_config.max_new_tokens = tokens
        
        chatbot = gr.Chatbot(label="Fetch Rewards Chatbot")
        temperature = gr.Slider(0, 1, value=0.6, step=0.1, label="Creativity", interactive=True)
        temperature.change(fn=update_temp, inputs=temperature)

        tokens = gr.Slider(50, 200, value=100, step=50, label="Length", interactive=True)
        tokens.change(fn=update_tokens, inputs=tokens)

        msg = gr.Textbox(label="", placeholder="Ask anything about Fetch!")
        clear = gr.Button("Clear Log")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            
            message = history[-1][0]
            prompt = f"""
            <human>: {message}
            <assistant>:
            """.strip()
            
            result = pipeline(
                prompt,
                generation_config=generation_config,
            )
            # print(result)
            parsed_result = result[0]["generated_text"].split("<assistant>:")[1][1:].split("\n")[0]
                
            history[-1][1] = ""
            for character in parsed_result:
                history[-1][1] += character
                time.sleep(0.01)
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()