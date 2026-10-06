import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 1: our "knowledge" — a few SAP policy sentences (pretend chunks) ---
chunks = [
    "Accounts for terminated employees must be locked within 24 hours.",   # 0
    "Initial passwords must be one-time passwords changed at first logon.", # 1
    "A user who can both create a vendor and pay it has an SoD conflict.",  # 2
    "The security audit log SM20 records failed logon attempts.",          # 3
]

# --- Step 2: the user's question (note: DIFFERENT words than any chunk) ---
question = "When is access removed for staff who have left the organisation?"

# --- Step 3: turn everything into embedding vectors ---
chunk_vectors = model.encode(chunks)      # shape: (4, 384) -> 4 vectors of 384 numbers
question_vector = model.encode(question)  # shape: (384,)   -> 1 vector of 384 numbers

print("Each chunk vector has", len(chunk_vectors[0]), "numbers.")
print("First 5 numbers of chunk 0:", np.round(chunk_vectors[0][:5], 3))
print("First 5 numbers of the question:", np.round(question_vector[:5], 3))
print()


# --- Step 4: cosine similarity = how close two vectors point (meaning closeness) ---
def cosine_similarity(a, b):
    # dot product divided by the product of the vectors' lengths (norms).
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print(f"QUESTION: {question}\n")
print("Similarity of the question to each chunk (higher = closer in meaning):\n")

scores = []
for i, chunk in enumerate(chunks):
    score = cosine_similarity(question_vector, chunk_vectors[i])
    scores.append((score, chunk))

# Sort highest-first so the best match is on top.
scores.sort(reverse=True)
for score, chunk in scores:
    print(f"  {score:.3f}   {chunk}")

print(f"\n>>> BEST MATCH (what RAG would send to Claude): {scores[0][1]}")
