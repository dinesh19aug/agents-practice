from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
import http.client
import json


from llama_index.core.workflow import Context
#d56ca0b17c893f4d6d6dbac00c8d049a5a673b83
def get_llm():
    return Ollama(
        model="qwen2.5:latest",
        temperature=0.7,
        max_tokens=200,
        request_timeout=120,
    )

async def web_search(query: str):
    """Useful for using the web to answer questions."""
    print(f"Web search for: {query}")
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": "apple inc"
        })
    headers = {
        'X-API-KEY': 'd56ca0b17c893f4d6d6dbac00c8d049a5a673b83',
        'Content-Type': 'application/json'
        }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    return res.read()


async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic. Your input should be notes with a title to save the notes under."""
    print(f"Recording notes: {notes_title}")
    async with ctx.store.edit_state() as ctx_state:
        if "research_notes" not in ctx_state["state"]:
            ctx_state["state"]["research_notes"] = {}
        ctx_state["state"]["research_notes"][notes_title] = notes
    return "Notes recorded."

async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic. Your input should be a markdown formatted report."""
    print(f"Writing report: {report_content}")
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["report_content"] = report_content
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback. Your input should be a review of the report."""
    print(f"Reviewing report: {review}")
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["review"] = review
    return "Report reviewed."

