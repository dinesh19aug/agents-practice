from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent


def create_llm():
    llm = Ollama(
        model="qwen2.5:latest",
        temperature=0.7,
        max_tokens=200,
        request_timeout=60,
        provider="auto")
    return llm



def create_workflow_agent(llm):
    print("Creating workflow agent with tools...")
    tool_spec = YahooFinanceToolSpec()
    workflow = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=llm,
        system_prompt="You are an agent that can fetch stock prices, news using tools.",
    )
    print("Workflow agent created successfully.")
    return workflow

async def main():
    print("Hello, world!")
    llm = create_llm()
    workflow = create_workflow_agent(llm)
    response = await workflow.run(user_msg="What is the latest news about PLTR?")
    print(response)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())