# --------------------------------------------
# First Ollama Demo File
# Accessing Ollama locally using Python
# Model: llama3:latest
# pip install ollama
# --------------------------------------------

import ollama

# Generate a response from a locally downloaded model
response = ollama.generate(
    model="llama3:latest",   # pick one of your downloaded models
    prompt="why summer is so hot these days"
)

# Print the raw response object
print(response)  # This will provide a dictionary-like object with keys like 'model', 'created_at', and 'response'

# Print the type of the response
print(type(response))  # This will show <class 'dict'> because Ollama returns a Python dictionary
