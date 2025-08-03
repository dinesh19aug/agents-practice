import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
load_dotenv()



llm = Ollama(
    model="qwen3:latest",
    temperature=0.7,
    max_tokens=100,
    request_timeout=120,
    provider="auto"
)

response = llm.complete("Who is Paul Graham?")
print(response)