import json

import rag

# --- Part 1: look at how embeddings are stored in index.json ---
chunks = rag.load_index("index.json")

print("=" * 60)
print("PART 1: How embeddings are stored")
print("=" * 60)

first_chunk = chunks[0]
print("Keys in a chunk:", list(first_chunk.keys()))
print("Type of 'embedding':", type(first_chunk["embedding"]))
print("Length of 'embedding':", len(first_chunk["embedding"]))
print("First 8 numbers:", first_chunk["embedding"][:8])
print()

# --- Part 2: run a real search and look at how scores are stored ---
print("=" * 60)
print("PART 2: How search results (scores) are stored")
print("=" * 60)

question = "What happens to a user account if someone enters the wrong password too many times?"
results = rag.search(question, chunks, top_k=3)

print("Type of 'results':", type(results))
print("Number of results:", len(results))
print()

for i, r in enumerate(results):
    print(f"--- result {i} ---")
    print("Keys:", list(r.keys()))
    print("score:", r["score"], "  (type:", type(r["score"]), ")")
    print("source:", r["source"])
    print("text snippet:", r["text"][:100], "...")
    print()
