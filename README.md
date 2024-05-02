# FAQ Bot

In this project, I finetuned an open-source LLM (Falcon) on web-scraped FAQ data from a website called Fetch. The purpose was to showcase how LLMs can provided contextual responses based on the data that it is finetuned on. For example, asking a question such as "Who is Wes?" may not give an appropriate response related to the company without finetuning.

## Libraries, Frameworks, APIs, Cloud Services

1. Libraries and Frameworks
	- Requests
	- BeautifulSoup
	- Pandas
	- Torch
	- Transformers
	- Bitsandbytes
	- Gradio
2. Resources
	- Fetch FAQ (https://fetch.com/faq#Receipts)

## How it works and services involved

This script performs the following tasks:

1. Scrapes FAQs from the Fetch website using BeautifulSoup.
2. Loads the Falcon LLM model using the Hugging Face transformers library.
3. Applies LoRA (Low-Rank Adaptation) for optimized fine-tuning of the model.
4. Fine-tunes the LLM model using the scraped FAQs data.
5. Creates a chatbot interface using Gradio that generates responses to user input based on the fine-tuned model.

### Demo

![](https://github.com/cdy3870/FAQ-BOT/blob/main/demo.gif)