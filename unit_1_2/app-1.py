import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smolagents import CodeAgent,tool,FinalAnswerTool,LiteLLMModel
import datetime
import requests
import pytz
import yaml
import json
from typing import List, Dict, Any, Optional

from unit_1_2.myOllama import OllamaModel, test_ollama_connection

# Test Ollama connection before proceeding
print("Testing Ollama connection...")
if not test_ollama_connection("qwen2.5:latest"):
    print("Please ensure Ollama is running and qwen2.5:latest model is pulled before continuing.")
    exit(1)

# Create Ollama model instance instead of HfApiModel
"""
model = OllamaModel(
    model_name="llava:latest",  # Use the exact model name from ollama list
    host="http://localhost:11434",
    max_tokens=4096,
    temperature=0.5
)
"""
model=LiteLLMModel(model_id="ollama/llava:latest", api_key="ollama",flatten_messages_as_text=True)

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Dinesh & Deepti Catering Co.": 4.9,
        "Dinesh Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)

    return f"The highest-rated catering service in Gotham City is {best_service}."

agent = CodeAgent(model=model, tools=[catering_service_tool, FinalAnswerTool()]
                  ,verbosity_level=1
                  ,max_steps=6,
                  )
agent.visualize()
agent.planning_interval = 2
result = agent.run(
    "Can you give me the name of the highest-rated catering service in Gotham City? Always use the output from tools and do not invent new answers.")


print("Result:", result)
