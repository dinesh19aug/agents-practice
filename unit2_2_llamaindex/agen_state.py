from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer


def get_llm():
    return Ollama(
        model="qwen2.5:latest",
        temperature=0.7,
        max_tokens=200,
        
    )

def get_agent(llm):
    workflow = FunctionAgent(
        llm=llm,
        tools=[set_name],
        system_prompt="You are an agent that can perform basic mathematical operations using tools.",
        verbose=False)
    print("Workflow agent created successfully.")
    return workflow

async def set_name(ctx: Context, name: str, age: int) -> str:
    state = await ctx.get("state")
    state["name"] = name,
    state["age"] = age
    await ctx.set("state", state)
    return f"Name set to {name}"

async def main():
    print("Hello, world!")
    llm = get_llm()
    workflow = get_agent(llm)
    
    print(f"*" * 20 ) 
    print("Using context to maintain state...")
    print (f"*" * 20 )
    ctx = Context(workflow=workflow)
    ctx = Context(workflow)

    # check if it knows a name before setting it
    response = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(str(response))

    # set the name using a tool
    response2 = await workflow.run(user_msg="My name is Dinesh. I am 43 years old.", ctx=ctx)
    print(str(response2))

    # retrieve the value from the state directly
    state = await ctx.get("state")
    print("Age as stored in state: ",state["age"])



    response = await workflow.run(user_msg="Hi My name is Dinesh. I am 43 years old.", ctx=ctx)
    print(response)
    response2 = await workflow.run(user_msg="What's my name? and how old will I be next year?", ctx=ctx)
    # retrieve the value from the state directly
    
    ## This should complan that it does not remember the name
    print(response2)
    state = await ctx.get("state")
    print("Age as stored in state: ",state["age"])
    
    print(f'{"*" * 20}' ) 
    print("Using context JsonSerializer...")
    print (f"*" * 20 )

    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    restored_ctx = Context.from_dict(
        workflow, ctx_dict, serializer=JsonSerializer()
        )
    
    response3 = await workflow.run(user_msg="What's my address?", ctx=restored_ctx)
    print(response3)
    print(f"ctx.store : {ctx.store}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())