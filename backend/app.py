"""
Luma Hair Care — support bot backend (routed multi-branch agent).

This version routes each turn to a specialist branch instead of running a
single agent loop. An intent classifier picks a lane, then the matching
specialist handles the turn:

    START -> classify_intent -> route_intent -> {
        product_advisor  <-> product_tools           (RAG over the KB)
        returns_agent    <-> returns_tools           (order lookup / policy)
                         -> human_approval -> apply_refund / deny_refund
                                                       (human-in-the-loop refund)
        general_response                              (greetings / fallback)
    } -> END

- `classify_intent` does a cheap single-label LLM classification and sticks to
  the returns lane while a refund flow is open.
- `product_advisor` is the grounded RAG behavior: bound with `[search_products]`
  (a Bedrock Knowledge Base retriever) and loops with `product_tools`.
- `returns_agent` follows a strict returns procedure with the store tools and
  emits `request_refund` as a signal that triggers the human-approval flow.
- `human_approval` calls `interrupt()` so a manager approves or denies before
  any refund is applied; `apply_refund` / `deny_refund` close the loop.
- An `InMemorySaver` checkpointer keyed by the request's `thread_id` gives the
  bot conversation memory and is required for `interrupt()` to pause/resume.

Run:
  uv run uvicorn app:app --reload --port 8000

Requires:
  - AWS credentials configured locally (e.g. `aws configure`)
  - Amazon Bedrock model access enabled in your region for MODEL_ID
  - A provisioned Bedrock Knowledge Base (KNOWLEDGE_BASE_ID)
"""
import json
import os
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrockConverse
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

import store

load_dotenv()

# Current Anthropic models on Bedrock require a cross-region inference profile
# id (prefixed by your region's geo: us.* / eu.* / apac.*). Override via env.
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")

# Now that the bot can retrieve Luma's real docs, tell it to ground its answers
# in what the tool returns instead of inventing details from pre-trained memory.
SYSTEM_PROMPT = (
    "You are Luma, the friendly customer support assistant for Luma Hair Care, a hair-care "
    "e-commerce brand.\n\n"
    "GROUNDING RULES (critical):\n"
    "- For ANY question about products, ingredients, routines, prices, or store policies, you MUST "
    "call the `search_products` tool first, and answer ONLY using the text it returns.\n"
    "- Do NOT use your own background knowledge about hair-care ingredients or products. If an "
    "ingredient, claim, price, or other detail is not present in the retrieved text, do not state it.\n"
    "- When listing a product's ingredients, list exactly the ones named in the retrieved text — never "
    "add, rename, expand, or guess additional ingredients.\n"
    "- If the retrieved text does not clearly answer the question, say you are not certain and suggest "
    "contacting the Luma team rather than guessing.\n\n"
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
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 6}},
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


# ---- Returns tools: wrap store.py for the returns agent ------------------


@tool
def lookup_order(order_number: str) -> str:
    """Look up a Luma Hair Care order by its order number. Returns customer
    name, email, items, order status, and refund status. Use this when the
    customer provides an order number and you need to see what they ordered."""
    try:
        order = store.get_order(order_number)
        if order is None:
            return f"No order found with number '{order_number}'. Please ask the customer to double-check."
        items_lines = []
        for item in order.get("items", []):
            items_lines.append(f"  - {item['product']} (qty {item['qty']}, ${item['price']:.2f} each)")
        items_str = "\n".join(items_lines) if items_lines else "  (no items)"
        return (
            f"Order {order_number}:\n"
            f"  Customer: {order.get('customer_name', 'N/A')} ({order.get('email', 'N/A')})\n"
            f"  Status: {order.get('status', 'unknown')}\n"
            f"  Purchased: {order.get('purchased', 'unknown')}\n"
            f"  Refund status: {order.get('refund_status', 'none')}\n"
            f"  Items:\n{items_str}"
        )
    except Exception as e:
        return f"Error looking up order '{order_number}': {e}"


@tool
def check_return_policy(order_number: str) -> str:
    """Check whether an order is eligible for a refund under Luma's 30-day
    return policy. Returns eligibility status and the reason. Call this after
    looking up the order to determine if a refund can proceed."""
    try:
        result = store.return_policy(order_number)
        eligible = result["eligible"]
        reason = result["reason"]
        days = result["days_since_purchase"]
        window = result["window_days"]
        status = "ELIGIBLE" if eligible else "NOT ELIGIBLE"
        return (
            f"Return policy check for order {order_number}:\n"
            f"  Status: {status}\n"
            f"  Reason: {reason}\n"
            f"  Days since purchase: {days}\n"
            f"  Return window: {window} days"
        )
    except Exception as e:
        return f"Error checking return policy for order '{order_number}': {e}"


@tool
def check_refund_status(order_number: str) -> str:
    """Check the current refund status of an order. Use this when a customer
    asks about the status of an existing refund request."""
    try:
        status = store.refund_status(order_number)
        if status is None:
            return f"No order found with number '{order_number}'. Cannot check refund status."
        return f"Refund status for order {order_number}: {status}"
    except Exception as e:
        return f"Error checking refund status for order '{order_number}': {e}"


@tool
def request_refund(order_number: str, amount: float, reason: str) -> str:
    """Request a refund for a customer's order. Call this ONLY after you have:
    1) Looked up the order and confirmed it exists,
    2) Verified the return policy shows the order is eligible,
    3) Confirmed the customer's identity (name or email matches the order),
    4) Stated the refund amount to the customer and they agreed.

    This tool signals that a refund is verified and ready for manager approval.
    It does NOT execute the refund — a human manager must approve first.

    Args:
        order_number: The order number to refund.
        amount: The total refund amount in dollars.
        reason: Brief reason for the refund (e.g. "Customer unsatisfied with product").
    """
    # This is a signal tool — it will be intercepted by the router and trigger
    # the human-approval flow. It should never actually execute via ToolNode.
    return (
        f"Refund request submitted for order {order_number}: "
        f"${amount:.2f} — reason: {reason}. Awaiting manager approval."
    )


# ---- Specialist LLMs (tools bound per-branch) ----------------------------
product_advisor_llm = llm.bind_tools([search_products])
# Force a knowledge-base lookup on a fresh product question so answers are grounded
# in retrieved docs (the model otherwise sometimes skips the tool and hallucinates).
product_advisor_llm_forced = llm.bind_tools([search_products], tool_choice="search_products")
returns_agent_llm = llm.bind_tools([lookup_order, check_return_policy, check_refund_status, request_refund])

# ---- ToolNodes for each branch -------------------------------------------
# product_tools handles the RAG retrieval tool for the product advisor branch.
product_tools = ToolNode([search_products])

# returns_tools handles the 3 normal returns tools. request_refund is
# intentionally excluded — it's a signal tool intercepted by route_returns
# to trigger the human-approval flow, never executed by a ToolNode.
returns_tools = ToolNode([lookup_order, check_return_policy, check_refund_status])

# Reject oversized inputs to bound per-request input-token cost / abuse.
MAX_INPUT_CHARS = 2000


# ---- The graph: an agent loop (chatbot <-> tools) -----------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[str]           # "product_advice" | "returns" | "general"
    active_flow: Optional[str]      # "returns" while a refund flow is open
    pending_refund: Optional[dict]  # approval payload captured from request_refund
    refund_applied: bool            # idempotency guard


# ---- Intent classification -----------------------------------------------

CLASSIFY_SYSTEM_PROMPT = (
    "You are an intent classifier for Luma Hair Care's support bot. "
    "Reply with EXACTLY ONE word and nothing else: product_advice, returns, or general.\n\n"
    "product_advice — anything about Luma's products: recommendations, ingredients, routines, "
    "hair types, how to use a product, AND prices, cost, sizes, or availability. Any question "
    "about a specific Luma product or the catalog is product_advice.\n"
    "returns — returning a product, refunds, order problems, or checking refund/return status.\n"
    "general — greetings, thanks, or anything unrelated to Luma products or returns.\n\n"
    "Examples:\n"
    "\"what's the price of the scalp reset serum?\" -> product_advice\n"
    "\"is the curl cream safe for color-treated hair?\" -> product_advice\n"
    "\"what products do you sell?\" -> product_advice\n"
    "\"I want a refund for order LUMA-1001\" -> returns\n"
    "\"has my refund gone through?\" -> returns\n"
    "\"hello\" -> general"
)


def _no_tool_messages(messages: list) -> list:
    """Drop tool-call and tool-result messages.

    `classify_intent` and `general_response` invoke the model WITHOUT a
    toolConfig, so any tool_use/tool_result blocks in the history make Bedrock
    Converse complain (and langchain-aws convert them to text with a warning).
    We remove BOTH halves of each tool exchange — the assistant tool-call message
    AND its ToolMessage result — so the remaining human/assistant text history
    stays valid and balanced.
    """
    cleaned = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if getattr(m, "tool_calls", None):
            continue
        cleaned.append(m)
    return cleaned


def classify_intent(state: State) -> dict:
    """Classify the user's intent. Short-circuits to 'returns' if a returns flow
    is already active and unresolved. Otherwise a cheap single-label LLM
    classification into product_advice | returns | general. Emits no user text."""
    if state.get("active_flow") == "returns" and not state.get("refund_applied"):
        return {"intent": "returns"}

    messages = [("system", CLASSIFY_SYSTEM_PROMPT)] + _no_tool_messages(state["messages"])
    response = llm.invoke(messages)
    # content may be a plain string OR a list of content blocks (Bedrock returns
    # blocks on the streaming path) — normalize either form with _to_text.
    raw = _to_text(response.content).strip().lower()
    if "return" in raw or "refund" in raw:
        intent = "returns"
    elif "product" in raw:
        intent = "product_advice"
    else:
        intent = "general"

    updates: dict = {"intent": intent}
    if intent == "returns":
        updates["active_flow"] = "returns"
    return updates


# ---- Routing function ----------------------------------------------------

def route_intent(state: State) -> str:
    """Route to the appropriate branch based on classified intent."""
    intent = state.get("intent", "general")
    mapping = {
        "product_advice": "product_advisor",
        "returns": "returns_agent",
        "general": "general_response",
    }
    return mapping.get(intent, "general_response")


# ---- General/fallback response node -------------------------------------

GENERAL_SYSTEM_PROMPT = (
    "You are Luma, the friendly customer support assistant for Luma Hair Care. "
    "The user's message doesn't clearly relate to a product question or a return/refund. "
    "Respond helpfully and concisely. If you're unsure what they need, ask a brief "
    "clarifying question. Keep the tone warm and conversational."
)


def general_response(state: State) -> dict:
    """Handle general/fallback messages — a plain reply that can ask a
    clarifying question if the user's intent is unclear."""
    messages = [("system", GENERAL_SYSTEM_PROMPT)] + _no_tool_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# ---- Product advisor node ------------------------------------------------

def product_advisor(state: State) -> dict:
    """The product-advice specialist. Grounds answers in the Bedrock Knowledge
    Base. On a fresh question we FORCE a `search_products` call so the model can't
    answer from memory; once the retrieved results (a ToolMessage) are in state,
    we let it answer normally."""
    messages = [("system", SYSTEM_PROMPT)] + state["messages"]
    last = state["messages"][-1] if state["messages"] else None
    # If the most recent message is NOT a tool result, we haven't retrieved yet
    # this turn -> force the search. After the ToolMessage comes back, answer freely.
    model = product_advisor_llm if isinstance(last, ToolMessage) else product_advisor_llm_forced
    return {"messages": [model.invoke(messages)]}


# ---- Returns agent node --------------------------------------------------

RETURNS_SYSTEM_PROMPT = (
    "You are Luma's returns specialist. You can ONLY act through tools — you cannot see orders, "
    "decide eligibility, or issue refunds on your own.\n\n"
    "WHAT YOU CAN OFFER:\n"
    "- The ONLY outcome you can offer is a FULL refund of the order total, and only after a human "
    "manager approves it. There are NO partial refunds, NO store credit, and NO return shipping "
    "labels. Never offer or mention any of these.\n"
    "- You do NOT decide eligibility yourself. Eligibility is exactly what the `check_return_policy` "
    "tool returns (an order qualifies if it was delivered, is within the 30-day window, and has not "
    "already been refunded). Do NOT invent rules about opened or used products, and do NOT ask how "
    "much of a product was used — none of that is relevant.\n\n"
    "HARD RULES:\n"
    "- NEVER say a refund has been processed, issued, approved, or completed. Calling `request_refund` "
    "only SUBMITS it to a manager for approval. After you call it, tell the customer the request has "
    "been submitted for manager approval.\n"
    "- NEVER state an order's details, eligibility, or refund amount unless a tool gave them to you. "
    "Do not invent policy, amounts, or next steps, and never tell the customer to contact customer "
    "service — you are the support agent handling this.\n\n"
    "PROCEDURE — call the real tools in order; do NOT describe these steps instead of doing them:\n"
    "1. If you don't have the order number, ask for it.\n"
    "2. Call `lookup_order` to fetch the order.\n"
    "3. Light identity check: confirm the name or email the customer gives matches the order record.\n"
    "4. Call `check_return_policy`.\n"
    "5. If it is NOT eligible, tell the customer why (use the tool's reason) and stop — do not call "
    "`request_refund`.\n"
    "6. If it IS eligible, tell the customer the full refund amount and ask them to confirm and give a "
    "brief reason.\n"
    "7. When they confirm, you MUST call `request_refund` with order_number, amount, and reason. "
    "Calling the tool is the ONLY way to start the refund — never just describe it in words.\n\n"
    "Keep replies warm and concise."
)


def returns_agent(state: State) -> dict:
    """The returns/refund specialist. Follows a strict procedure: gather order
    number → look up → identity check → policy check → state amount → request
    refund. Bound with all 4 returns tools (including request_refund as a signal
    tool that triggers the human-approval flow via route_returns)."""
    messages = [("system", RETURNS_SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [returns_agent_llm.invoke(messages)]}


# ---- Returns routing function --------------------------------------------

def route_returns(state: State) -> str:
    """Route the returns agent's output:
    - If the model called request_refund → route to human_approval (HITL)
    - If the model called other tools → route to returns_tools for execution
    - If the model gave a plain reply (no tool calls) → END this turn
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if any(c["name"] == "request_refund" for c in calls):
        return "human_approval"
    if calls:
        return "returns_tools"
    return "__end__"


# ---- Human-in-the-loop refund nodes --------------------------------------


def human_approval(state: State) -> dict:
    """Pause the graph for manager approval of a refund request.

    Extracts the request_refund call args, looks up the order for customer/item
    context, then calls interrupt() to surface the approval payload to the client.
    On resume, the manager's decision is recorded into pending_refund.
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    refund_call = next((c for c in calls if c["name"] == "request_refund"), None)

    order_number = refund_call["args"]["order_number"]
    amount = refund_call["args"]["amount"]
    reason = refund_call["args"]["reason"]
    tool_call_id = refund_call["id"]

    # Get customer info and items from the order
    order = store.get_order(order_number)
    customer = order["customer_name"] if order else "Unknown"
    items = order.get("items", []) if order else []

    # Build the pending_refund payload
    pending = {
        "order_number": order_number,
        "customer": customer,
        "items": items,
        "amount": amount,
        "reason": reason,
        "tool_call_id": tool_call_id,
    }

    # Interrupt — this pauses the graph and sends the payload to the client
    decision = interrupt({
        "type": "refund_approval",
        "order_number": order_number,
        "customer": customer,
        "items": items,
        "amount": amount,
        "reason": reason,
        "instructions": "Approve or deny this refund.",
    })

    # When resumed, decision contains {"approved": bool, "reason": ..., "reviewer": ...}
    return {
        "pending_refund": {
            **pending,
            "approved": bool(decision.get("approved")),
            "decision_reason": decision.get("reason"),
            "reviewer": decision.get("reviewer"),
        }
    }


def apply_refund(state: State) -> dict:
    """Apply the approved refund — idempotent (no-op if already applied).

    Double guard: checks both the state flag (refund_applied) and the data layer
    (store.refund_status) before mutating. Appends a ToolMessage answering the
    original request_refund tool call to satisfy the message-protocol contract.
    """
    pr = state["pending_refund"]
    if not state.get("refund_applied") and store.refund_status(pr["order_number"]) != "refunded":
        store.apply_refund(pr["order_number"])
    tool_msg = ToolMessage(
        content=f"Refund of ${pr['amount']:.2f} processed for order {pr['order_number']}.",
        tool_call_id=pr["tool_call_id"],
    )
    return {"messages": [tool_msg], "refund_applied": True, "active_flow": None}


def deny_refund(state: State) -> dict:
    """Handle a denied refund — no data change, just communicate the denial.

    Appends a ToolMessage answering the original request_refund tool call with
    the denial reason so the message-protocol contract is satisfied.
    """
    pr = state["pending_refund"]
    reason_text = pr.get("decision_reason") or "No reason provided"
    tool_msg = ToolMessage(
        content=f"Refund denied for order {pr['order_number']}. Reason: {reason_text}",
        tool_call_id=pr["tool_call_id"],
    )
    return {"messages": [tool_msg], "active_flow": None}


def route_after_approval(state: State) -> str:
    """Route based on the manager's decision: approved → apply, else → deny."""
    return "apply_refund" if state["pending_refund"]["approved"] else "deny_refund"


# ---- The graph: a routed multi-branch agent ------------------------------
# START -> classify_intent -> route_intent -> specialist branch -> END.
# product_advisor / returns_agent loop with their ToolNodes; the returns
# branch can detour through human_approval -> apply_refund / deny_refund.

builder = StateGraph(State)
builder.add_node("classify_intent", classify_intent)
builder.add_node("product_advisor", product_advisor)
builder.add_node("product_tools", product_tools)  # existing ToolNode([search_products])
builder.add_node("returns_agent", returns_agent)
builder.add_node("returns_tools", returns_tools)  # existing ToolNode([...])
builder.add_node("human_approval", human_approval)
builder.add_node("apply_refund", apply_refund)
builder.add_node("deny_refund", deny_refund)
builder.add_node("general_response", general_response)

builder.add_edge(START, "classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    ["product_advisor", "returns_agent", "general_response"],
)

# tools_condition returns "tools" | "__end__"; remap "tools" -> our renamed node.
builder.add_conditional_edges(
    "product_advisor",
    tools_condition,
    {"tools": "product_tools", "__end__": END},
)
builder.add_edge("product_tools", "product_advisor")

builder.add_conditional_edges(
    "returns_agent",
    route_returns,
    ["returns_tools", "human_approval", "__end__"],
)
builder.add_edge("returns_tools", "returns_agent")

builder.add_conditional_edges(
    "human_approval",
    route_after_approval,
    ["apply_refund", "deny_refund"],
)
builder.add_edge("apply_refund", "returns_agent")
builder.add_edge("deny_refund", "returns_agent")

builder.add_edge("general_response", END)

# Memory: persist each thread's message history so follow-up questions have
# context, and so interrupt() can pause and resume the refund flow. The front
# end sends a stable thread_id per browser session.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ---- Web layer ----------------------------------------------------------
app = FastAPI(title="Luma Support Bot — Returns Agent (routing + HITL refunds)")

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


class ResumeRequest(BaseModel):
    """A manager's decision on a paused refund, sent to POST /resume."""
    thread_id: str
    approved: bool
    reason: Optional[str] = None
    reviewer: Optional[str] = None


# Nodes whose token output should be streamed to the user. Internal nodes
# (classifier, ToolNodes, approval plumbing) are filtered out so the client
# never sees routing labels or raw retrieved chunks.
USER_FACING_NODES = {"product_advisor", "returns_agent", "general_response"}


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


def sse(event: str, data: str) -> str:
    """Format a single Server-Sent Event frame.

    The client switches on the event name: `token` (append text), `interrupt`
    (render approval panel), `done` (turn finished), `error` (something failed).
    Multi-line payloads are emitted as multiple `data:` lines per the SSE spec so
    a newline inside a token can't corrupt the frame; the client rejoins them
    with "\\n".
    """
    data_lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{data_lines}\n"


# Headers that keep SSE flowing: disable client/proxy buffering so frames are
# delivered as soon as they're yielded.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _tool_label(call: dict) -> dict:
    """Human-friendly label for a tool call, surfaced to the client for visibility."""
    name = call.get("name", "tool")
    args = call.get("args", {}) or {}
    order = args.get("order_number")
    labels = {
        "search_products": "Searching the product knowledge base",
        "lookup_order": f"Looking up order {order}" if order else "Looking up the order",
        "check_return_policy": f"Checking the return policy for {order}" if order else "Checking the return policy",
        "check_refund_status": f"Checking refund status for {order}" if order else "Checking refund status",
        "request_refund": "Submitting the refund for manager approval",
    }
    return {"name": name, "label": labels.get(name, f"Calling {name}")}


def _stream_graph(inputs, config):
    """Stream a graph run as SSE frames.

    Forwards text from user-facing nodes as `token` events, surfaces a paused
    refund as an `interrupt` event, and always closes with `done`. Any failure
    is reported as an `error` event so the client can recover gracefully.
    """
    try:
        for mode, payload in graph.stream(
            inputs,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") in USER_FACING_NODES:
                    text = _to_text(chunk.content)
                    if text:
                        yield sse("token", text)
            elif mode == "updates":
                for node_name, update in payload.items():
                    if node_name == "__interrupt__":
                        intr = update[0]
                        yield sse("interrupt", json.dumps(intr.value))
                        continue
                    # Surface the tools an agent node is about to call (visibility).
                    if node_name in USER_FACING_NODES and isinstance(update, dict):
                        for msg in (update.get("messages") or []):
                            for call in (getattr(msg, "tool_calls", None) or []):
                                yield sse("tool", json.dumps(_tool_label(call)))
        yield sse("done", "[DONE]")
    except Exception as e:  # noqa: BLE001 — surface any failure to the client
        yield sse("error", str(e))


@app.get("/")
def health():
    return {
        "status": "ok",
        "stage": "4-returns-hitl",
        "model": MODEL_ID,
        "region": AWS_REGION,
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Stream the graph's reply as Server-Sent Events.

    The graph routes to a specialist branch which may pause to run tools or to
    request manager approval of a refund. We forward text from the user-facing
    nodes as `token` events, emit an `interrupt` event when the refund flow
    pauses, and finish with `done`. Guard rejections are emitted over SSE too so
    the frontend handles every response uniformly.
    """
    message = (req.message or "").strip()

    # A stable thread_id lets the checkpointer recall this conversation. Fall
    # back to a shared default if the client didn't send one.
    thread_id = (req.thread_id or "").strip() or "default"
    config = {"configurable": {"thread_id": thread_id}}

    # Input guards: emit the rejection as a token + done so the client renders
    # it like any other reply, then short-circuit before touching the graph.
    if not message:
        def reject_empty():
            yield sse("token", "Please type a message.")
            yield sse("done", "[DONE]")
        return StreamingResponse(
            reject_empty(), media_type="text/event-stream", headers=SSE_HEADERS
        )
    if len(message) > MAX_INPUT_CHARS:
        def reject_long():
            yield sse("token", "That message is a bit long — please shorten it and try again.")
            yield sse("done", "[DONE]")
        return StreamingResponse(
            reject_long(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    return StreamingResponse(
        _stream_graph({"messages": [("user", message)]}, config),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/resume")
def resume(req: ResumeRequest):
    """Resume a paused refund flow with a manager's decision.

    The graph paused at `human_approval` via interrupt(); resuming injects the
    decision with Command(resume=...) and streams the continuation as SSE. If
    nothing is interrupted for this thread, returns a clear error.
    """
    thread_id = (req.thread_id or "").strip() or "default"
    config = {"configurable": {"thread_id": thread_id}}

    # Guard: only resume when the graph is actually paused at human_approval.
    # get_state().next is empty when the run is finished and won't contain the
    # approval node otherwise.
    state = graph.get_state(config)
    if "human_approval" not in (state.next or ()):
        def no_pause():
            yield sse("error", "Nothing is awaiting approval for this conversation.")
        return StreamingResponse(
            no_pause(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    decision = {
        "approved": req.approved,
        "reason": req.reason,
        "reviewer": req.reviewer,
    }
    return StreamingResponse(
        _stream_graph(Command(resume=decision), config),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
