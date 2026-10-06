# SAP Policy RAG Bot

A terminal-based Retrieval-Augmented Generation (RAG) assistant that answers questions about SAP security policy documents (user administration, segregation of duties, change management) — grounded in the actual policy text, with source citations and multi-turn conversation memory.

This is Project 2 of a hands-on AI Engineering portfolio. It follows on from [Project 1](../project1-sap-chatbot), a plain Claude chatbot — this project adds real retrieval over a private knowledge base, which is the core pattern behind most production LLM applications.

---

## Why RAG, and why this architecture

A general-purpose LLM has no idea what *your* company's actual SAP policies say — it can only guess based on what's typical. RAG fixes that: before answering, the bot searches your real policy documents for the most relevant passages and hands those to Claude as context, instructing it to answer only from that context and cite its source. The result is an assistant that's actually trustworthy for compliance-type questions, instead of one that sounds confident but might be wrong.

**Key architecture decision — local embeddings.** The original plan was to use OpenAI's `text-embedding-3-small` for the retrieval step. In practice, the IBM API gateway used for this project only exposes chat models (Claude, GPT, Gemini, etc.) — no embedding models at all. Rather than depending on an external embedding API, this project embeds entirely **locally** using `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dimensional, ~90MB, runs offline). This is actually a stronger security story for SAP data: the policy text never leaves the machine during the embedding step, only the final retrieved snippets are sent to Claude for answering.

---

## Features

- **Semantic search over real documents** — retrieves the most relevant policy chunks by meaning, not just keyword match (confirmed working on a question with zero shared keywords with the matching text)
- **Source-cited answers** — every claim is attributed to the policy file it came from
- **Anti-hallucination guardrail** — explicitly instructed to say "not in the context" rather than guess, verified on out-of-scope questions
- **Multi-turn conversation memory** — clean question/answer history persists for the session; retrieval re-runs fresh each turn so follow-ups ("what about X on top of that?") work correctly without resending old context forever
- **Streaming responses** — replies appear live, token by token
- **Robust error handling** — automatic retries with exponential backoff on transient API errors (network, rate limit, server overload)
- **Token usage logging** — per-turn and cumulative session totals
- **Secure config** — API keys loaded from `.env`, never hard-coded

---

## Tech stack

- **Python 3**
- **Anthropic Claude API** (`anthropic` SDK) — via a configurable base URL, for answering
- **sentence-transformers** (`all-MiniLM-L6-v2`) — for local embeddings
- **pypdf** — for reading PDF policy documents
- **numpy** — for cosine similarity scoring
- **python-dotenv** — environment/secret management

---

## Setup

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd project2-sap-rag-bot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS / Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and fill in your real values:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ANTHROPIC_BASE_URL=https://your-base-url-here
   ```

5. **Build the index and run it**
   ```bash
   python main.py
   ```
   The first run downloads the local embedding model (~90MB) and builds `index.json` from the documents in `data/`.

---

## Usage

Type your SAP policy question at the `You:` prompt. Type `exit` or `quit` to end the session.

Example:
```
You: How quickly must access be removed after someone leaves the company?
Assistant: Access must be revoked within 24 hours of termination (Source: user_administration_policy.txt).
[tokens] this turn: 834 in / 543 out | session total: 834 in / 543 out
```

---

## Project structure

```
project2-sap-rag-bot/
├── main.py             # Chat loop: retrieval + Claude + streaming/retry/memory
├── rag.py               # Core RAG pipeline (load, chunk, embed, search, build_prompt)
├── inspect_index.py     # Standalone script to inspect the built index
├── data/                 # Source SAP policy documents (.txt / .pdf)
├── requirements.txt    # Pinned dependencies
├── .env.example         # Template for API keys (copy to .env)
├── .gitignore            # Keeps secrets, venv, and the generated index out of Git
└── README.md
```

---

## How it works (pipeline)

```
Load docs -> Chunk (sliding window, 120 words, 20 overlap) -> Embed (local model)
   -> Save index.json -> [per question] Search (cosine similarity, top_k=3)
   -> Build prompt (labeled context + anti-hallucination instruction) -> Claude -> Answer with citations
```

## Concepts demonstrated

Document chunking (sliding window with overlap) · text embeddings and vector representations · cosine similarity for semantic search · retrieval-augmented generation (RAG) · prompt engineering for grounding and citation · local vs. API-based embedding models as an architecture tradeoff · multi-turn conversation design with separated "stored" vs. "sent" context · streaming API responses · resilient error handling with exponential backoff · token usage/cost tracking.

---

## What's next

Part of a 5-project AI Engineering portfolio. Next up: Project 3.
