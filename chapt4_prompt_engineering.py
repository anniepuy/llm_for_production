"""
Title: Chapt 4 - Prompt engineering
Author: Ann Hagan
Date: 2-27-2025
overview: Example prompt engineering for a language model
"""

import ollama

#1. Zero-shot prompting is when the model is asked to produce output based instructions only
def generate_poem():
    prompt_system = """You are a helpful assistant whose goal is to write short poems"""
    prompt = """Write a short poem about the ocean"""
    
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt},
    ]
    
    response = ollama.chat(
        model='llama3.2', 
        messages=messages
    )
    
    return response['message']['content']

#2. Few-shot prompting is when the model is given a few examples to perform the task 
def generate_few_shot():  # Fixed function name for consistency
    prompt_system = """You are a helpful assistant whose goal is to write short poems"""
    
    examples = "\n".join([
        "Example 1:\nBirston fills the air,\nMountains high and valley deep,\nNature's sweet music.",
        "Example 2:\nDesert sands so hot,\nMirage of water so near,\nDesert's cruel trick."
    ])
    
    prompt = f"""Write a short poem about the ocean in the same style as these examples:\n{examples}"""
    
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
    
    response = ollama.chat(
        model='llama3.2',  # Fixed model name
        messages=messages
    )
    
    return response['message']['content']

if __name__ == "__main__":
    try:
        # Run zero-shot generation
        print("\n=== Zero-shot Generated Poem ===")
        poem = generate_poem()
        print(poem)
        
        # Run few-shot generation
        print("\n=== Few-shot Generated Poem ===")
        few_shot_poem = generate_few_shot()
        print(few_shot_poem)
        
    except Exception as e:
        print(f"Error: {e}")
