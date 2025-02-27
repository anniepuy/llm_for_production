"""
Title: chapt_1
Author: Ann Hagan
Date: 1-15-2025
Purpose: Chatper 1 of BUilding LLMs for Production exercise code. Basic chatbot script to interact with Ollama Mistral and few-shot learning.
"""

#Note, the book uses OpenAi's API call. This code uses Ollama Mistral locally
import requests
import ollama
import logging

#set up logger
# Configure the logger
logging.basicConfig(
    filename='app_log.txt',  # Log file name
    level=logging.DEBUG,     # Logging level (DEBUG captures all levels)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    filemode='w'             # Overwrite the log file each time; use 'a' to append
)

logger = logging.getLogger(__name__)
#english text to translate
english_text = "Hello, how are you?"
logger.debug(f"English text to translate: {english_text}")

messages = [
    {"role": "system",
     "content": "You are a helpful assistant."},
    {"role": "user",
     "content": f'''Translate the following English text to French: "{english_text}"'''},
    
]

response = ollama.chat(model='mistral',
                          messages=messages,)
logger.debug(f"Response from Ollama mistral: {response}")

print(response["message"]["content"])

"""
Controlling LLMs with Few-Shot Learning
This provides the LLM a few examples before making predictions.
This helps the LLM discover patterns in the data and make better predictions.
"""
#prompt for summarization
prompt = """ Describe the following movies using animals: {movie}"""

#few-shot examples
examples = [ 
    {"input": "Titantic", "output": "Orca, Penquin, Starfish"},
    {"input": "The Matrix", "output": "Cougar, Chameleon, Black Panther"},
]

#Sending the examples to the llm and asking the question for Toy Story
movie = "Toy Story"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt.format(movie=examples[0]["input"])},
    {"role": "assistant", "content": examples[0]["output"]},
    {"role": "user", "content": prompt.format(movie=examples[1]["input"])},
    {"role": "assistant", "content": examples[1]["output"]},
    {"role": "user", "content": prompt.format(movie=movie)},
]
response = ollama.chat(model='mistral', 
                       messages=messages)

print(response["message"]["content"])
logger.debug(f"Response from Ollama mistral: {prompt} {response}")

