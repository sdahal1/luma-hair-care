# Requirements Document

**Feature:** Agentic Returns Assistant (Refund-Only First Pass)

## Introduction

Luma's support bot today is a single-purpose agent: it answers product, ingredient,
routine, and policy questions by retrieving from a Bedrock Knowledge Base, with
streaming responses and per-thread conversation memory. The graph is a simple
agent loop (`chatbot ⇄ tools`) with one tool (`search_products`).

This feature upgrades the bot from "answers questions" to "takes actions." It adds
the capabilities that justify LangGraph over a plain LangChain chain:

1. **Intent-based routing** — classify each turn and route to a specialized branch
   (product advice vs. returns/finance) instead of one catch-all prompt.
2. **Multi-step reasoning with tools** — a refund request is decomposed into
   discrete steps: verify the order, check the return/refund policy, compute the
   refund, and (after approval) apply it.
3. **Human-in-the-loop (HITL) refund approval** — before any refund is finalized,
   the graph pauses (`interrupt()`), a store manager approves or denies, and the
   graph resumes from the exact checkpoint.

**Scope for this first pass: refund-only.** Product *exchanges* (inventory checks
for a replacement and shipping-label generation) are intentionally deferred to a
follow-up so this pass stays focused on the HITL refund pattern.

To keep the demo low-overhead, all order/refund data is **mocked in a local JSON
file** behind tool functions — no database. Customer identity is resolved entirely
from the **order number** the user provides; there is no login or separate identity
system.

**Frontend constraint:** the storefront stays exactly as it is — plain HTML, CSS,
and vanilla JS (`index.html`, `styles.css`, `app.js`). No framework, no build step,
no new dependencies. Frontend changes for this feature must be minimal and additive.
The focus of this build is the LangGraph implementation, not the UI.

This is a demo/learning build (it backs a blog series). "Production-grade" is a
non-goal; clarity of the LangGraph patterns is the priority. Existing product Q&A
must keep working unchanged.

## Glossary

- **Intent**: the category of a user turn — `product_advice`, `returns`, or `general`.
- **Returns branch**: the specialized part of the graph that handles refunds (and,
  in a later pass, exchanges).
- **Order record**: the mock data entry keyed by order number; it carries the
  customer name/email, line items, status, and refund status.
- **HITL interrupt**: a paused graph execution awaiting a manager decision.
- **Resume**: continuing a paused execution after a manager decision, keyed by `thread_id`.

## Requirements

### Requirement 1: Preserve existing product Q&A

**User story:** As a shopper, I want to keep asking product, ingredient, routine,
and policy questions and get grounded answers, so that the new features don't
regress what already works.

#### Acceptance Criteria

1. WHEN a user sends a product/ingredient/routine/policy question THEN the system SHALL route it to the product-advice branch and answer using the `search_products` knowledge-base retrieval, as it does today.
2. WHEN the product-advice branch answers THEN the system SHALL stream the reply token-by-token over the existing `/chat` response.
3. WHEN the retrieved context does not cover the question THEN the system SHALL say it is unsure and suggest contacting the Luma team rather than inventing details.
4. THE system SHALL preserve per-thread conversation memory so follow-up questions retain context.

### Requirement 2: Intent-based routing

**User story:** As a support operator, I want each message routed to a focused
specialist branch, so that the bot's behavior is reliable and easy to reason about.

#### Acceptance Criteria

1. WHEN a new user turn arrives THEN the system SHALL classify its intent as one of `product_advice`, `returns`, or `general`.
2. IF the intent is `product_advice` THEN the system SHALL route to the product-advice branch (Requirement 1).
3. IF the intent is `returns` THEN the system SHALL route to the returns branch (Requirement 4).
4. IF the intent is `general` or cannot be confidently classified THEN the system SHALL route to a general/fallback response that can ask a clarifying question.
5. WHEN an ongoing returns workflow is mid-execution for a thread THEN the system SHALL continue that workflow rather than re-classifying the turn from scratch.
6. THE routing decision SHALL be made by an explicit routing step (conditional edge), not hidden inside an unrelated node.

### Requirement 3: Order lookup and identity from order number

**User story:** As a customer starting a return, I want to identify my purchase by
order number, so that the bot can see my order without me logging in.

#### Acceptance Criteria

1. WHEN the returns branch needs order context AND no order number is present in the conversation THEN the system SHALL ask the user for their order number.
2. WHEN the user provides an order number THEN the system SHALL look it up via the `lookup_order` tool against the mock data.
3. IF the order number is not found THEN the system SHALL tell the user it could not find that order and ask them to re-check the number.
4. WHEN an order is found THEN the system SHALL derive the customer's name and purchased items from the order record (the order number is the identity anchor).
5. WHEN a refund is about to be finalized THEN the system SHALL perform a light identity check by confirming a detail on the order (e.g. the name or email) matches what the user states.

### Requirement 4: Multi-step refund processing

**User story:** As a customer, I want to request a refund and have the bot work
through the necessary checks, so that the request is handled end-to-end.

#### Acceptance Criteria

1. WHEN a refund request is in progress THEN the system SHALL check the order against the return/refund policy (e.g. within the returns window, eligible status) using retrieved policy info and/or order fields.
2. IF the order is not eligible for a refund THEN the system SHALL explain why (e.g. outside the returns window) and not proceed to approval.
3. WHEN the order is eligible THEN the system SHALL compute the refund amount from the order's line items and present it to the user before requesting approval.
4. WHEN the user asks about the status of an existing refund THEN the system SHALL report it via the `check_refund_status` tool.
5. THE refund branch SHALL execute these as distinct, single-purpose steps so each tool call is observable and testable.
6. IF a tool call fails THEN the system SHALL surface a graceful message and not crash the conversation.

> **Note:** product exchanges (replacement inventory checks via `check_inventory`
> and shipping-label generation via `create_shipping_label`) are deferred to a
> follow-up pass and are out of scope here.

### Requirement 5: Human-in-the-loop refund approval

**User story:** As a store manager, I want to approve or deny refunds before money
moves, so that the bot can't autonomously issue an invalid or expensive refund.

#### Acceptance Criteria

1. WHEN a refund is ready to be finalized THEN the system SHALL pause execution using a LangGraph `interrupt()` before any refund side effect occurs.
2. WHEN the graph pauses for approval THEN the system SHALL surface an approval payload containing at least the order number, customer, item(s), refund amount, and reason.
3. WHILE awaiting a decision THE system SHALL persist the paused state via the checkpointer, keyed by the thread's `thread_id`.
4. WHEN a manager approves THEN the system SHALL resume from the checkpoint, apply the refund to the mock data (set refund status), and confirm completion to the customer.
5. WHEN a manager denies THEN the system SHALL resume, NOT change the refund status, and communicate the denial (with reason if provided) to the customer.
6. THE refund-applying step SHALL be idempotent so that a resume or replay does not issue the refund twice.
7. IF no decision has been provided THEN the system SHALL NOT finalize the refund.

### Requirement 6: Mock data layer (no database)

**User story:** As the developer, I want order/refund data mocked in a local file,
so that I can demo the agent without provisioning a database.

#### Acceptance Criteria

1. THE system SHALL store mock orders in a local JSON file loaded by the backend at startup.
2. THE order records SHALL be keyed by order number and SHALL include customer name, email, line items (product, qty, price), order status, refund status, and purchase date.
3. WHEN a tool mutates state (e.g. applying a refund) THEN the system SHALL update the in-memory representation so the change is visible for the rest of the session.
4. THE mock data products SHALL be consistent with the existing catalog in `app.js` / the knowledge base.
5. THE data-access layer SHALL be isolated behind tool functions so it can later be swapped for a real API without changing the graph.
6. THE data file MAY include an inventory map for the deferred exchange feature, but the refund-only pass SHALL NOT depend on it.

### Requirement 7: Backend API support for pause/resume

**User story:** As the frontend, I need an API that can carry a paused approval back
and forth, so that the HITL refund works over HTTP.

#### Acceptance Criteria

1. THE `/chat` endpoint SHALL continue to accept `{ message, thread_id }` and stream normal replies.
2. WHEN graph execution pauses on an interrupt THEN the API SHALL signal to the client that an approval is pending, including the approval payload and the `thread_id`.
3. THE system SHALL expose a way to submit a manager decision (approve/deny, with optional reason/reviewer) for a given `thread_id` and resume execution.
4. WHEN a decision is submitted THEN the API SHALL resume the graph from its checkpoint and stream the post-approval reply.
5. THE input-size and basic abuse guards that exist today SHALL remain in place.

### Requirement 8: Manager approval UI

**User story:** As a store manager using the demo, I want a simple way to see and
act on a pending refund, so that the HITL step is visible end-to-end.

#### Acceptance Criteria

1. WHEN an approval is pending for the conversation THEN the UI SHALL display the approval details (order, customer, item, amount, reason) with Approve and Deny actions.
2. WHEN the manager clicks Approve or Deny THEN the UI SHALL submit the decision for the conversation's `thread_id` and then render the resumed bot reply.
3. WHILE a refund is awaiting approval THE UI SHALL make clear to the customer that the request is pending review.
4. THE approval control SHALL be implemented in the existing vanilla JS (`app.js`) as a lightweight inline affordance (e.g. a simple panel with two buttons) — no framework, no build step, no new frontend dependencies.
5. THE frontend changes SHALL be additive and minimal, preserving the current look and structure of the storefront and chat widget.

## Out of scope

- **Product exchanges** for this pass: replacement inventory checks (`check_inventory`)
  and shipping-label generation (`create_shipping_label`). Deferred to a follow-up.
- Real authentication, accounts, or a real customer-identity service.
- A real database or real payment/refund processing.
- Real carrier/shipping integration.
- A frontend framework, build step, or new frontend dependencies — the UI stays
  vanilla HTML/CSS/JS and minimal.
- Production deployment, multi-region, or durable cross-restart persistence
  (in-memory checkpointer is acceptable for the demo).
- Notifying managers through real channels (email/SMS/Slack); a demo affordance is enough.
