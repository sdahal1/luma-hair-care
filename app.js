/* ============================================================
   Luma Hair Care — front-end logic
   - Renders the product grid + product detail modal
   - Controls the support chat widget

   LangGraph readiness (single-feature demo: product Q&A over a
   Bedrock Knowledge Base, with memory + streaming):
   - Every chat request carries a persistent `thread_id` so the
     LangGraph checkpointer can remember the conversation.
   - `sendToBackend()` consumes a streamed response (SSE or raw
     text) and emits tokens as they arrive, so the widget can
     render the reply live. A simulated stream keeps the UX
     visible until CHAT_API_URL is wired up.
   ============================================================ */

/* ---------- Config ---------- */
// The Stage-1 LangGraph/FastAPI backend (raw Bedrock model, no RAG).
// Set back to null to restore the offline placeholder reply.
const CHAT_API_URL = "http://localhost:8000/chat";
// Resume a paused human-in-the-loop refund: a manager's approve/deny decision
// is POSTed here with the conversation's thread_id, then the graph continues
// streaming the post-decision reply over the same SSE contract.
const RESUME_API_URL = "http://localhost:8000/resume";

/* ---------- Product data ----------
   These fields double as the source content for the Bedrock
   Knowledge Base (see /knowledge-base/*.md). Keep them in sync. */
const PRODUCTS = [
  {
    id: "define-curl-cream",
    name: "Define Curl Cream",
    desc: "Lightweight curl-defining cream for bouncy, frizz-free coils.",
    price: "$24",
    size: "8 fl oz",
    rating: "4.9 ★",
    reviews: 1840,
    tag: "Bestseller",
    color: "#b5654a",
    hairTypes: ["Curly", "Coily", "Wavy"],
    keyIngredients: ["Shea Butter", "Flaxseed Extract", "Glycerin"],
    usage: "Rake through soaking-wet hair, scrunch upward, then air-dry or diffuse.",
    safety: "Color-safe. Sulfate-, silicone-, and paraben-free. Vegan & cruelty-free.",
  },
  {
    id: "repair-bond-oil",
    name: "Repair Bond Oil",
    desc: "Strengthening treatment oil that rebuilds damaged, over-processed hair.",
    price: "$32",
    size: "3.4 fl oz",
    rating: "4.8 ★",
    reviews: 1210,
    tag: "New",
    color: "#2f6f5f",
    hairTypes: ["Damaged", "Color-treated", "Chemically processed"],
    keyIngredients: ["Bond-building Complex", "Argan Oil", "Vitamin E"],
    usage: "Apply 2–3 drops to damp mid-lengths and ends before styling, or to dry hair as a finisher.",
    safety: "Color-safe and bleach-safe. Sulfate- and paraben-free. Vegan & cruelty-free.",
  },
  {
    id: "hydra-wash-shampoo",
    name: "Hydra Wash Shampoo",
    desc: "Sulfate-free hydrating shampoo that cleanses without stripping moisture.",
    price: "$22",
    size: "10 fl oz",
    rating: "4.7 ★",
    reviews: 2050,
    tag: null,
    color: "#7b9bd1",
    hairTypes: ["Dry", "Normal", "Color-treated"],
    keyIngredients: ["Aloe Vera", "Coconut-derived Cleansers", "Panthenol"],
    usage: "Massage into wet scalp, lather, and rinse. Follow with Silk Slip Conditioner.",
    safety: "Color-safe, sulfate-free, pH-balanced. Vegan & cruelty-free.",
  },
  {
    id: "silk-slip-conditioner",
    name: "Silk Slip Conditioner",
    desc: "Deeply nourishing conditioner for easy detangling and lasting softness.",
    price: "$23",
    size: "10 fl oz",
    rating: "4.9 ★",
    reviews: 1760,
    tag: null,
    color: "#c79bd1",
    hairTypes: ["Dry", "Curly", "Color-treated"],
    keyIngredients: ["Slip-rich Conditioning Blend", "Shea Butter", "Hydrolyzed Quinoa"],
    usage: "Work through clean, wet hair, comb to detangle, leave 1–2 minutes, then rinse.",
    safety: "Color-safe, silicone-free. Vegan & cruelty-free.",
  },
  {
    id: "scalp-reset-serum",
    name: "Scalp Reset Serum",
    desc: "Soothing pre-wash serum that balances the scalp and supports healthy growth.",
    price: "$28",
    size: "2 fl oz",
    rating: "4.6 ★",
    reviews: 690,
    tag: "New",
    color: "#d1a94f",
    hairTypes: ["Oily scalp", "Sensitive scalp", "All types"],
    keyIngredients: ["Niacinamide", "Tea Tree", "Peptide Blend"],
    usage: "Massage into a dry or damp scalp 10 minutes before washing. Use 2–3 times a week.",
    safety: "Fragrance-free option available. Sulfate- and paraben-free. Vegan & cruelty-free.",
  },
  {
    id: "air-dry-styling-gel",
    name: "Air-Dry Styling Gel",
    desc: "Flexible-hold gel for soft, crunch-free definition on wash-and-go days.",
    price: "$20",
    size: "8 fl oz",
    rating: "4.7 ★",
    reviews: 980,
    tag: null,
    color: "#d18b6f",
    hairTypes: ["Wavy", "Curly", "Coily"],
    keyIngredients: ["Flaxseed Gel", "Marshmallow Root", "Glycerin"],
    usage: "Smooth over wet hair in sections, scrunch, and air-dry. Scrunch out any cast when dry.",
    safety: "Color-safe, alcohol-free, no flaking. Vegan & cruelty-free.",
  },
  {
    id: "weekly-repair-mask",
    name: "Weekly Repair Mask",
    desc: "Intensive deep-conditioning mask for dry, frizzy, and damaged hair.",
    price: "$30",
    size: "8 fl oz",
    rating: "4.8 ★",
    reviews: 1430,
    tag: "Bestseller",
    color: "#6a8f7d",
    hairTypes: ["Dry", "Frizzy", "Damaged", "Color-treated"],
    keyIngredients: ["Murumuru Butter", "Hyaluronic Acid", "Rice Protein"],
    usage: "Apply to clean, damp hair once a week, leave 5–10 minutes, then rinse well.",
    safety: "Color-safe, silicone-free. Vegan & cruelty-free.",
  },
  {
    id: "heat-shield-mist",
    name: "Heat Shield Mist",
    desc: "Weightless thermal protection up to 450°F before heat styling.",
    price: "$19",
    size: "5 fl oz",
    rating: "4.6 ★",
    reviews: 870,
    tag: null,
    color: "#9a8fd1",
    hairTypes: ["All types", "Fine", "Color-treated"],
    keyIngredients: ["Heat-activated Polymers", "Bamboo Extract", "Provitamin B5"],
    usage: "Mist evenly over damp or dry hair before blow-drying, flat-ironing, or curling.",
    safety: "Color-safe, non-sticky, lightweight. Vegan & cruelty-free.",
  },
];

function renderProducts() {
  const grid = document.getElementById("product-grid");
  if (!grid) return;
  grid.innerHTML = PRODUCTS.map((p) => `
    <article class="product-card" data-product="${p.id}" tabindex="0" role="button" aria-label="View details for ${p.name}">
      <div class="product-media" style="background: linear-gradient(160deg, #f6e3d8, #efe7f3);">
        ${p.tag ? `<span class="product-tag">${p.tag}</span>` : ""}
        <div class="product-bottle" style="--bottle:${p.color}"></div>
      </div>
      <div class="product-body">
        <h3 class="product-name">${p.name}</h3>
        <p class="product-desc">${p.desc}</p>
        <div class="product-meta">
          <span class="product-price">${p.price}</span>
          <span class="product-rating">${p.rating}</span>
        </div>
        <button class="product-add" data-add="${p.name}">Add to cart</button>
      </div>
    </article>
  `).join("");
}

/* ---------- Product detail modal ---------- */
const Modal = {
  el: null,
  lastFocused: null,

  ensure() {
    if (this.el) return this.el;
    const el = document.createElement("div");
    el.className = "modal-overlay";
    el.id = "product-modal";
    el.setAttribute("aria-hidden", "true");
    el.innerHTML = `
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <button class="modal-close" aria-label="Close details">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18M6 6l12 12"/></svg>
        </button>
        <div class="modal-body"></div>
      </div>`;
    document.body.appendChild(el);

    el.addEventListener("click", (e) => { if (e.target === el) this.close(); });
    el.querySelector(".modal-close").addEventListener("click", () => this.close());
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && el.classList.contains("is-open")) this.close();
    });
    this.el = el;
    return el;
  },

  open(productId) {
    const p = PRODUCTS.find((x) => x.id === productId);
    if (!p) return;
    const el = this.ensure();
    this.lastFocused = document.activeElement;

    el.querySelector(".modal-body").innerHTML = `
      <div class="modal-hero">
        <div class="modal-bottle" style="--bottle:${p.color}"></div>
        ${p.tag ? `<span class="product-tag">${p.tag}</span>` : ""}
      </div>
      <div class="modal-info">
        <h2 id="modal-title">${p.name}</h2>
        <div class="modal-meta">
          <span class="product-price">${p.price}</span>
          <span class="modal-size">${p.size}</span>
          <span class="product-rating">${p.rating} (${p.reviews.toLocaleString()})</span>
        </div>
        <p class="modal-desc">${p.desc}</p>

        <div class="modal-section">
          <h4>Best for</h4>
          <div class="chips">${p.hairTypes.map((h) => `<span class="chip">${h}</span>`).join("")}</div>
        </div>
        <div class="modal-section">
          <h4>Key ingredients</h4>
          <div class="chips">${p.keyIngredients.map((i) => `<span class="chip">${i}</span>`).join("")}</div>
        </div>
        <div class="modal-section">
          <h4>How to use</h4>
          <p>${p.usage}</p>
        </div>
        <div class="modal-section">
          <h4>Good to know</h4>
          <p>${p.safety}</p>
        </div>

        <div class="modal-actions">
          <button class="btn btn-primary" data-add="${p.name}">Add to cart — ${p.price}</button>
          <button class="btn btn-ghost" data-ask="${p.name}">Ask the assistant about this</button>
        </div>
      </div>`;

    el.classList.add("is-open");
    el.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    setTimeout(() => el.querySelector(".modal-close").focus(), 50);
  },

  close() {
    if (!this.el) return;
    this.el.classList.remove("is-open");
    this.el.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    if (this.lastFocused) this.lastFocused.focus();
  },
};

/* ---------- Cart (tiny demo behavior) ---------- */
function setupCart() {
  const countEl = document.querySelector(".cart-count");
  let count = 0;
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-add]");
    if (!btn) return;
    count += 1;
    if (countEl) countEl.textContent = String(count);
    const original = btn.textContent;
    btn.textContent = "Added ✓";
    btn.disabled = true;
    setTimeout(() => { btn.textContent = original; btn.disabled = false; }, 1200);
  });
}

/* ---------- Product interactions (open modal, ask assistant) ---------- */
function setupProductInteractions() {
  document.addEventListener("click", (e) => {
    // "Ask the assistant about this" — opens chat with a prefilled question
    const askBtn = e.target.closest("[data-ask]");
    if (askBtn) {
      const name = askBtn.getAttribute("data-ask");
      Modal.close();
      Chat.open();
      Chat.prefill(`Is the ${name} right for my hair type?`);
      return;
    }
    // Add-to-cart shouldn't open the modal
    if (e.target.closest("[data-add]")) return;

    // Open product detail modal from a card
    const card = e.target.closest("[data-product]");
    if (card) Modal.open(card.getAttribute("data-product"));
  });

  // Keyboard: Enter/Space opens the focused product card
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const card = e.target.closest("[data-product]");
    if (card) { e.preventDefault(); Modal.open(card.getAttribute("data-product")); }
  });
}

/* ============================================================
   Chat widget
   ============================================================ */
const Chat = {
  panel: null, launcher: null, messages: null, form: null,
  input: null, suggestions: null, isOpen: false, greeted: false,

  init() {
    this.panel = document.getElementById("chat-panel");
    this.launcher = document.getElementById("chat-launcher");
    this.messages = document.getElementById("chat-messages");
    this.form = document.getElementById("chat-form");
    this.input = document.getElementById("chat-text");
    this.suggestions = document.getElementById("chat-suggestions");
    if (!this.panel) return;

    this.launcher.addEventListener("click", () => this.toggle());
    document.getElementById("chat-close").addEventListener("click", () => this.close());

    document.querySelectorAll("[data-open-chat]").forEach((el) => {
      el.addEventListener("click", (e) => { e.preventDefault(); this.open(); });
    });

    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = this.input.value.trim();
      if (text) this.send(text);
    });

    this.suggestions.addEventListener("click", (e) => {
      const chip = e.target.closest(".suggestion-chip");
      if (chip) this.send(chip.textContent.trim());
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.isOpen) this.close();
    });
  },

  toggle() { this.isOpen ? this.close() : this.open(); },

  open() {
    this.isOpen = true;
    this.panel.classList.add("is-open");
    this.panel.setAttribute("aria-hidden", "false");
    this.launcher.classList.add("is-open");
    this.launcher.setAttribute("aria-expanded", "true");
    if (!this.greeted) {
      this.greeted = true;
      this.addBot("Hi there! 👋 I'm the Luma support assistant. Ask me about products, ingredients, or what's right for your hair type.");
    }
    setTimeout(() => this.input.focus(), 250);
  },

  close() {
    this.isOpen = false;
    this.panel.classList.remove("is-open");
    this.panel.setAttribute("aria-hidden", "true");
    this.launcher.classList.remove("is-open");
    this.launcher.setAttribute("aria-expanded", "false");
  },

  prefill(text) {
    this.input.value = text;
    setTimeout(() => this.input.focus(), 300);
  },

  scrollToBottom() { this.messages.scrollTop = this.messages.scrollHeight; },

  addUser(text) {
    const el = document.createElement("div");
    el.className = "msg msg-user";
    el.textContent = text;
    this.messages.appendChild(el);
    this.scrollToBottom();
  },

  addBot(text) {
    const el = document.createElement("div");
    el.className = "msg msg-bot";
    setRichText(el, text || "");
    this.messages.appendChild(el);
    this.scrollToBottom();
    return el;
  },

  addToolNote(label) {
    const el = document.createElement("div");
    el.className = "msg-tool";
    el.textContent = "🔧 " + label;
    this.messages.appendChild(el);
    this.scrollToBottom();
    return el;
  },

  showTyping() {
    const el = document.createElement("div");
    el.className = "msg msg-bot msg-typing";
    el.innerHTML = "<span></span><span></span><span></span>";
    this.messages.appendChild(el);
    this.scrollToBottom();
    return el;
  },

  async send(text) {
    this.input.value = "";
    this.addUser(text);
    let typing = this.showTyping();
    let botEl = null;
    let interrupted = false;

    const clearTyping = () => { if (typing && typing.isConnected) typing.remove(); typing = null; };
    const ensureBot = () => { if (!botEl) { clearTyping(); botEl = this.addBot(""); } return botEl; };

    const callbacks = {
      onTool: (info) => {
        clearTyping();
        this.addToolNote(info && info.label ? info.label : "Working…");
        botEl = null;                 // next text starts a fresh bubble after the tool note
        typing = this.showTyping();   // keep a working indicator going
      },
      onToken: (token) => {
        const el = ensureBot();
        setRichText(el, (el.dataset.raw || "") + token);
        this.scrollToBottom();
      },
      onInterrupt: (payload) => {
        interrupted = true;
        clearTyping();
        this.renderApprovalPanel(payload);
      },
      onError: (message) => {
        clearTyping();
        if (botEl) botEl.remove();
        this.addBot(message || "Sorry, I hit a snag. Please try again in a moment.");
      },
    };

    try {
      const full = await sendToBackend(text, callbacks);
      clearTyping();
      if (!botEl && !interrupted && full) this.addBot(full);
    } catch (err) {
      clearTyping();
      if (botEl) botEl.remove();
      this.addBot("Sorry, I'm having trouble connecting right now. Please try again in a moment.");
      console.error(err);
    }
  },

  /* ---------- Human-in-the-loop refund approval ----------
     When the backend pauses on an interrupt, it sends an approval payload.
     We render an inline panel with the refund details and Approve / Deny
     buttons. Acting on either POSTs the decision to /resume and streams the
     continuation into a fresh bot bubble. */
  renderApprovalPanel(payload) {
    const panel = document.createElement("div");
    panel.className = "msg msg-bot approval-panel";

    const amount = (typeof payload.amount === "number")
      ? `$${payload.amount.toFixed(2)}`
      : `$${payload.amount ?? ""}`;
    const items = Array.isArray(payload.items) ? payload.items : [];
    const itemsHtml = items.length
      ? items.map((it) =>
          `<li>${escapeHtml(it.product)} <span class="approval-qty">× ${escapeHtml(it.qty)}</span></li>`
        ).join("")
      : "<li>(no items listed)</li>";

    panel.innerHTML = `
      <div class="approval-title">Refund approval needed</div>
      <dl class="approval-details">
        <div><dt>Order</dt><dd>${escapeHtml(payload.order_number || "")}</dd></div>
        <div><dt>Customer</dt><dd>${escapeHtml(payload.customer || "")}</dd></div>
        <div><dt>Amount</dt><dd>${escapeHtml(amount)}</dd></div>
        <div><dt>Reason</dt><dd>${escapeHtml(payload.reason || "")}</dd></div>
      </dl>
      <ul class="approval-items">${itemsHtml}</ul>
      <p class="approval-note">Awaiting manager approval…</p>
      <div class="approval-actions">
        <button class="btn btn-primary approval-approve" type="button">Approve</button>
        <button class="btn btn-ghost approval-deny" type="button">Deny</button>
      </div>`;

    this.messages.appendChild(panel);
    this.scrollToBottom();

    const actions = panel.querySelector(".approval-actions");
    const note = panel.querySelector(".approval-note");

    // Resolve the panel: remove the buttons (no double-submit), update the
    // pending note to reflect the decision, then resume the graph.
    const decide = (approved) => {
      actions.remove();
      note.textContent = approved
        ? "Approved — processing the refund…"
        : "Denied — letting the customer know…";
      this.resumeRefund(approved);
    };

    panel.querySelector(".approval-approve").addEventListener("click", () => decide(true));
    panel.querySelector(".approval-deny").addEventListener("click", () => decide(false));
    return panel;
  },

  async resumeRefund(approved) {
    let typing = this.showTyping();
    let botEl = null;
    let interrupted = false;

    const clearTyping = () => { if (typing && typing.isConnected) typing.remove(); typing = null; };
    const ensureBot = () => { if (!botEl) { clearTyping(); botEl = this.addBot(""); } return botEl; };

    const callbacks = {
      onTool: (info) => {
        clearTyping();
        this.addToolNote(info && info.label ? info.label : "Working…");
        botEl = null;
        typing = this.showTyping();
      },
      onToken: (token) => {
        const el = ensureBot();
        setRichText(el, (el.dataset.raw || "") + token);
        this.scrollToBottom();
      },
      onInterrupt: (payload) => {
        interrupted = true;
        clearTyping();
        this.renderApprovalPanel(payload);
      },
      onError: (message) => {
        clearTyping();
        if (botEl) botEl.remove();
        this.addBot(message || "Sorry, something went wrong resuming the refund.");
      },
    };

    try {
      const res = await fetch(RESUME_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: getThreadId(),
          approved: approved,
          reviewer: "manager-demo",
        }),
      });
      if (!res.ok) throw new Error(`Resume error: ${res.status}`);

      const contentType = res.headers.get("content-type") || "";
      const full = await readStream(res, contentType, callbacks);
      clearTyping();
      if (!botEl && !interrupted && full) this.addBot(full);
    } catch (err) {
      clearTyping();
      if (botEl) botEl.remove();
      this.addBot("Sorry, I couldn't submit that decision. Please try again.");
      console.error(err);
    }
  },
};

/* ---------- Conversation identity (memory) ----------
   A stable thread id lets the LangGraph checkpointer tie every
   message to one continuous conversation. Persisted so the
   "it remembers" behavior survives a page reload. */
function getThreadId() {
  const KEY = "luma_thread_id";
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : "t-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    localStorage.setItem(KEY, id);
  }
  return id;
}

/* ---------- Backend bridge ----------
   Sends { message, thread_id } and streams the reply back token by
   token. `callbacks` is an object of optional handlers:
     - onToken(text):      a streamed text fragment to append
     - onInterrupt(payload): a refund paused for manager approval
     - onError(message):   the backend reported an error event
   For convenience a bare function is accepted and treated as onToken.
   Handles SSE (typed `event:` frames) and raw text streams; falls back
   to a single JSON { reply }. Until CHAT_API_URL is set, a simulated
   stream keeps the UX visible. */
async function sendToBackend(text, callbacks) {
  if (typeof callbacks === "function") callbacks = { onToken: callbacks };
  callbacks = callbacks || {};
  const onToken = callbacks.onToken;
  const threadId = getThreadId();

  if (!CHAT_API_URL) {
    const msg = "Thanks for your message! Our AI assistant isn't live just yet — " +
      "it's being built with LangGraph and Amazon Bedrock, and it'll answer from our " +
      "real product knowledge base. In the meantime, email hello@lumahair.com and a human will help.";
    await simulateStream(msg, onToken);
    return msg;
  }

  const res = await fetch(CHAT_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text, thread_id: threadId }),
  });
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);

  const contentType = res.headers.get("content-type") || "";
  const isStream = res.body && (
    contentType.includes("text/event-stream") ||
    contentType.includes("application/x-ndjson") ||
    contentType.includes("text/plain")
  );
  if (isStream) return readStream(res, contentType, callbacks);

  const data = await res.json();
  const reply = data.reply ?? data.message ?? "(no response)";
  if (onToken) onToken(reply);
  return reply;
}

/* Read a streamed response, dispatching on the SSE `event:` type.
   The backend emits typed frames:
     event: token     data: <text fragment>
     event: interrupt  data: <JSON approval payload>
     event: done       data: [DONE]
     event: error      data: <message>
   `callbacks` may provide onToken / onInterrupt / onError. A non-SSE
   (raw text) response keeps the legacy behavior: every chunk is a token. */
async function readStream(res, contentType, callbacks) {
  callbacks = callbacks || {};
  const { onToken, onInterrupt, onError, onTool } = callbacks;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const isSSE = contentType.includes("text/event-stream");
  let full = "";
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });

    if (!isSSE) {
      // Raw-text stream fallback: every chunk is reply text.
      full += chunk;
      if (onToken) onToken(chunk);
      continue;
    }

    // Server-Sent Events: split on blank lines, then read the `event:` and
    // `data:` lines within each frame to decide how to handle it.
    buffer += chunk;
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      if (!frame.trim()) continue;
      let eventType = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) {
          let d = line.slice(5);
          if (d.startsWith(" ")) d = d.slice(1);  // strip only the single SSE separator space
          dataLines.push(d);
        }
      }
      const data = dataLines.join("\n");

      if (eventType === "interrupt") {
        let payload = null;
        try { payload = JSON.parse(data); }
        catch (e) { console.error("Bad interrupt payload:", e, data); }
        if (payload && onInterrupt) onInterrupt(payload);
      } else if (eventType === "error") {
        if (onError) onError(data);
        else throw new Error(data || "Stream error");
      } else if (eventType === "tool") {
        let info = null;
        try { info = JSON.parse(data); } catch (e) { info = { label: data }; }
        if (onTool) onTool(info);
      } else if (eventType === "done") {
        // End of this turn; nothing to append.
        continue;
      } else {
        // token (or an untyped frame for the raw-text fallback path).
        if (data === "[DONE]") continue;
        full += data;
        if (onToken) onToken(data);
      }
    }
  }
  return full;
}

// Simulated token stream for the placeholder (word by word).
async function simulateStream(text, onToken) {
  if (!onToken) return;
  const words = text.split(" ");
  for (let i = 0; i < words.length; i++) {
    onToken((i === 0 ? "" : " ") + words[i]);
    await new Promise((r) => setTimeout(r, 28));
  }
}

// Escape user/order-derived text before injecting into innerHTML so the
// approval panel can't be used to inject markup.
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[c]));
}

/* ---------- Lightweight markdown for bot replies ----------
   The model answers in markdown (**bold**, `-` bullets, numbered steps).
   Appending that with textContent shows the raw syntax, so we render a
   small, safe subset instead: bold, italic, inline code, bullet and
   numbered lists, and paragraphs.

   Safety: the text is HTML-escaped FIRST, then only our own tags are
   introduced. Model output is never treated as markup, so a reply
   containing <script> or an onerror= attribute stays inert text. */
function renderInline(escaped) {
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,!?;:])/g, "$1<em>$2</em>");
}

function renderRichText(raw) {
  const lines = escapeHtml(raw).split("\n");
  const out = [];
  let list = null;          // { tag: "ul"|"ol", items: [] }
  let para = [];            // buffered plain lines

  const flushPara = () => {
    if (para.length) { out.push("<p>" + renderInline(para.join("<br>")) + "</p>"); para = []; }
  };
  const flushList = () => {
    if (list) {
      out.push("<" + list.tag + ">" +
        list.items.map((i) => "<li>" + renderInline(i) + "</li>").join("") +
        "</" + list.tag + ">");
      list = null;
    }
  };

  for (const line of lines) {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (heading) {
      flushPara();
      flushList();
      out.push('<p class="msg-heading">' + renderInline(heading[1]) + "</p>");
    } else if (bullet || numbered) {
      flushPara();
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((bullet ? bullet[1] : numbered[1]));
    } else if (!line.trim()) {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return out.join("");
}

/* Set a bubble's text, keeping the raw markdown on the node so streamed
   tokens can be re-rendered as the reply grows (a token may land in the
   middle of a `**bold**` span, so we always re-render the whole reply). */
function setRichText(el, raw) {
  el.dataset.raw = raw;
  el.innerHTML = renderRichText(raw);
}

/* ---------- Boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  renderProducts();
  setupCart();
  setupProductInteractions();
  Chat.init();
});
