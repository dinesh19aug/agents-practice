
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

def create_llm():
    llm = Ollama(
        model="qwen2.5:latest",
        temperature=0.7,
        max_tokens=200,
        request_timeout=30,
        provider="auto")
    return llm

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product and double it"""
    print
    return a * b * 2


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum and adds extra 2. """
    print(f"Adding {a} and {b}")
    return a + b + 2

def create_workflow_agent(llm):
    print("Creating workflow agent with tools...")
    workflow = FunctionAgent(
        tools=[multiply, add],
        llm=llm,
        system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    )
    print("Workflow agent created successfully.")
    print(f"Available tools: {workflow.get_tools}" )
    return workflow


async def main():
    print("Hello, world!")
    llm=create_llm()
    workflow = create_workflow_agent(llm)
    response = await workflow.run(user_msg="What is 2+4?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())