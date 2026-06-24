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
// Point this at your LangGraph/FastAPI endpoint when it's ready.
// e.g. "http://localhost:8000/chat"
const CHAT_API_URL = null;

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
    el.textContent = text;
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
    const typing = this.showTyping();
    let botEl = null;

    const onToken = (token) => {
      if (!botEl) { typing.remove(); botEl = this.addBot(""); }
      botEl.textContent += token;
      this.scrollToBottom();
    };

    try {
      const full = await sendToBackend(text, onToken);
      if (!botEl) { typing.remove(); this.addBot(full); }
      else if (full && botEl.textContent !== full) { botEl.textContent = full; }
    } catch (err) {
      typing.remove();
      if (botEl) botEl.remove();
      this.addBot("Sorry, I'm having trouble connecting right now. Please try again in a moment.");
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
   token via onToken(). Handles SSE ("data:" frames) and raw text
   streams; falls back to a single JSON { reply }. Until
   CHAT_API_URL is set, a simulated stream keeps the UX visible. */
async function sendToBackend(text, onToken) {
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
  if (isStream) return readStream(res, contentType, onToken);

  const data = await res.json();
  const reply = data.reply ?? data.message ?? "(no response)";
  if (onToken) onToken(reply);
  return reply;
}

async function readStream(res, contentType, onToken) {
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
      full += chunk;
      if (onToken) onToken(chunk);
      continue;
    }

    // Server-Sent Events: parse "data:" lines, split on blank lines.
    buffer += chunk;
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        full += payload;
        if (onToken) onToken(payload);
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

/* ---------- Boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  renderProducts();
  setupCart();
  setupProductInteractions();
  Chat.init();
});
