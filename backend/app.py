"""
Luma Hair Care — support bot backend (STAGE 1: raw model, no RAG).

This is the "baby step" version for the video's "watch it fail" beat: a
single-node LangGraph graph that calls a Bedrock model directly with NO
tools, NO Knowledge Base, and NO memory. It answers only from the model's
pre-trained knowledge, so it will confidently invent details about Luma's
specific products (hallucinate). That failure is intentional — the next
stage adds a Bedrock Knowledge Base tool to ground it.

Run:
  uv pip install -r requirements.txt   (or: pip install -r requirements.txt)
  uvicorn app:app --reload --port 8000

Requires:
  - AWS credentials configured locally (e.g. `aws configure`)
  - Amazon Bedrock model access enabled in your region for MODEL_ID
"""
import os
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# Current Anthropic models on Bedrock require a cross-region inference profile
# id (prefixed by your region's geo: us.* / eu.* / apac.*). Override via env.
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# Intentionally minimal: a support persona, but NO product catalog. This makes
# the bot answer confidently from pre-trained knowledge and hallucinate about
# Luma's actual products. That visible failure is what motivates RAG next.
SYSTEM_PROMPT = (
    "You are Luma, the friendly and knowledgeable customer support assistant "
    "for Luma Hair Care, a hair-care e-commerce brand. Answer customer "
    "questions about products, ingredients, and routines helpfully and confidently."
)

# max_tokens caps the output length of every call, which bounds the
# per-request cost so a single reply can't run away.
llm = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=AWS_REGION,
    temperature=0.7,
    max_tokens=1024,
)

# Reject oversized inputs to bound per-request input-token cost / abuse.
MAX_INPUT_CHARS = 2000


# ---- The graph: State -> one node -> END --------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> dict:
    """The one and only node: send the messages to the model, return its reply."""
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()


# ---- Web layer ----------------------------------------------------------
app = FastAPI(title="Luma Support Bot — Stage 1 (raw model, no RAG)")

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
    thread_id: Optional[str] = None  # accepted now; used once we add memory


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
    return {"status": "ok", "stage": "1-raw-model", "model": MODEL_ID, "region": AWS_REGION}


@app.post("/chat")
def chat(req: ChatRequest):
    """Stream the model's reply token-by-token as plain text.

    No memory and no tools yet: each call starts fresh (so it also "forgets"),
    and answers are ungrounded (so it can hallucinate product details).
    """
    message = (req.message or "").strip()
    if not message:
        return PlainTextResponse("Please type a message.")
    if len(message) > MAX_INPUT_CHARS:
        return PlainTextResponse("That message is a bit long — please shorten it and try again.")

    def generate():
        messages = [("system", SYSTEM_PROMPT), ("user", message)]
        for chunk, _meta in graph.stream({"messages": messages}, stream_mode="messages"):
            text = _to_text(chunk.content)
            if text:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")
