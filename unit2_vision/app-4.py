from smolagents import CodeAgent, LiteLLMModel
from smolagents import tool
from smolagents.agents import ToolCallingAgent
from smolagents import DuckDuckGoSearchTool

model=LiteLLMModel(model_id="ollama_chat/llava:latest", api_key="ollama")

agent=CodeAgent(tools=[],model=model,
                add_base_tools=True,
                additional_authorized_imports=['numpy', 'sys','wikipedia','scipy','requests', 'bs4'])

agent.run("What is 2+2?")