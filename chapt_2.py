"""
Title: chapt_2
Author: Ann Hagan
Date: 1-15-2025
Purpose: Chatper 2 Attention in action
"""

#Modified not to use deepspeed for the  Attention in action exercise as deepspeed uses pydantic <2.0.0 and ollam required pydantic 2.9.0
#uses smaller model for M4 Mac. Must uninstall deepspeed to run this code


#Note, the book uses OpenAi's API call. This code uses Ollama Mistral locally
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set device to MPS (if available) or CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the model and tokenizer
model_name = "facebook/opt-125m"  # Smaller model for M4 Mac
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to(device)


# Input text
inp = "The quick brown fox jumps over the lazy dog."
inp_tokenized = tokenizer(inp, return_tensors="pt").to(device)

# Forward pass with attention outputs
with torch.no_grad():
    outputs = model(**inp_tokenized)

# Extract attention weights
attentions = outputs.attentions  # List of attention weights for each layer

print(f"Number of layers: {len(attentions)}")
print(f"Shape of attention weights in first layer: {attentions[0].shape}")
