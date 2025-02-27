"""
Title: Chapt 3 - Finding the Perplexity of a Language Model
Author: Ann Hagan
Date: 2-18-2025
overview: Simple script to find the perplexity of a language model. Perplexity value is encountered when the model  is determining probabilitues for sequences of words.
"""
import numpy as np

probabilities = np.array([0.1, 0.27, 0.55, 0.78])

sentence_probability = probabilities.prod()
sentance_probability_normalized = sentence_probability ** ( 1 / len(probabilities))
perpelexity = 1 / sentance_probability_normalized
print(perpelexity)