"""
Luma Hair Care — support bot backend (STAGE 3: RAG + tools + memory).

This version grounds the bot in Luma's real catalog using a Bedrock Knowledge
Base. The graph is now an agent loop:

    START -> chatbot -> (tools_condition) -> tools -> chatbot -> ... -> END

- `search_products` wraps a Bedrock Knowledge Base retriever. The model calls
  it for any product / ingredient / routine / policy question.
- `llm.bind_tools([...])` lets the model decide when to retrieve.
- `ToolNode` runs the tool; the conditional edge loops back so the model can
  read the retrieved context and answer (or call the tool again).
- An `InMemorySaver` checkpointer keyed by the request's `thread_id` gives the
  bot conversation memory, so follow-ups like "is it safe for my hair?" work.

Compared to Stage 1 (raw model, hallucinates) the only new capability that
matters for the demo is RAG — same model, now grounded.

Run:
  uv run uvicorn app:app --reload --port 8000

Requires:
  - AWS credentials configured locally (e.g. `aws configure`)
  - Amazon Bedrock model access enabled in your region for MODEL_ID
  - A provisioned Bedrock Knowledge Base (KNOWLEDGE_BASE_ID)
"""
import os
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Current Anthropic models on Bedrock require a cross-region inference profile
# id (prefixed by your region's geo: us.* / eu.* / apac.*). Override via env.
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")

# Now that the bot can retrieve Luma's real docs, tell it to ground its answers
# in what the tool returns instead of inventing details from pre-trained memory.
SYSTEM_PROMPT = (
    "You are Luma, the friendly customer support assistant for Luma Hair Care, "
    "a hair-care e-commerce brand. "
    "For any question about products, ingredients, routines, or store policies, "
    "ALWAYS call the search_products tool first and base your answer only on the "
    "information it returns. Do not invent product names, ingredients, prices, or "
    "policies. If the retrieved information doesn't cover the question, say you're "
    "not sure and suggest contacting the Luma team rather than guessing. "
    "Keep replies warm, concise, and helpful."
)

# max_tokens caps the output length of every call, which bounds the
# per-request cost so a single reply can't run away.
llm = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=AWS_REGION,
    temperature=0.7,
    max_tokens=1024,
)

# ---- RAG tool: search Luma's Bedrock Knowledge Base ----------------------
# The retriever does a vector search over the KB (Titan v2 embeddings backed by
# S3 Vectors) and returns the most relevant chunks from Luma's product docs,
# FAQ, and hair-type guide.
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=KNOWLEDGE_BASE_ID,
    region_name=AWS_REGION,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)


@tool
def search_products(query: str) -> str:
    """Search Luma Hair Care's knowledge base for product, ingredient, routine,
    or policy information. Use this for any question about Luma's catalog or
    store policies. `query` should be a natural-language search string."""
    docs = retriever.invoke(query)
    if not docs:
        return "No matching information found in the Luma knowledge base."
    return "\n\n---\n\n".join(d.page_content for d in docs)


tools = [search_products]
llm_with_tools = llm.bind_tools(tools)

# Reject oversized inputs to bound per-request input-token cost / abuse.
MAX_INPUT_CHARS = 2000


# ---- The graph: an agent loop (chatbot <-> tools) -----------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> dict:
    """Call the model (which may decide to call search_products).

    The system prompt is prepended here at invoke time rather than stored in
    state, so memory accumulates only the real conversation turns (and we never
    duplicate the system message across checkpointed turns).
    """
    messages = [("system", SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
# If the model emitted tool calls, route to "tools"; otherwise go to END.
builder.add_conditional_edges("chatbot", tools_condition)
# After running tools, loop back so the model can read the results and answer.
builder.add_edge("tools", "chatbot")

# Memory: persist each thread's message history so follow-up questions have
# context. The front end sends a stable thread_id per browser session.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ---- Web layer ----------------------------------------------------------
app = FastAPI(title="Luma Support Bot — Stage 3 (RAG + tools + memory)")

# Local-dev CORS. Tighten allow_origins to your real site origin before
# exposing this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


def _to_text(content) -> str:
    """Bedrock content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            parts.append(part.get("text", "") if isinstance(part, dict) else str(part))
        return "".join(parts)
    return str(content)


@app.get("/")
def health():
    return {
        "status": "ok",
        "stage": "3-rag",
        "model": MODEL_ID,
        "region": AWS_REGION,
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Stream the model's reply token-by-token as plain text.

    The graph may pause mid-stream to run the search_products tool, then resume
    streaming the grounded answer. We only forward text from the "chatbot" node
    so the user never sees raw retrieved chunks or tool-call plumbing.
    """
    message = (req.message or "").strip()
    if not message:
        return PlainTextResponse("Please type a message.")
    if len(message) > MAX_INPUT_CHARS:
        return PlainTextResponse("That message is a bit long — please shorten it and try again.")

    # A stable thread_id lets the checkpointer recall this conversation. Fall
    # back to a shared default if the client didn't send one.
    thread_id = (req.thread_id or "").strip() or "default"
    config = {"configurable": {"thread_id": thread_id}}

    def generate():
        for chunk, meta in graph.stream(
            {"messages": [("user", message)]},
            config=config,
            stream_mode="messages",
        ):
            # Only stream the assistant's text. Skip the tools node (retrieved
            # chunks) and any non-text blocks (tool-call arguments).
            if meta.get("langgraph_node") != "chatbot":
                continue
            text = _to_text(chunk.content)
            if text:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")
