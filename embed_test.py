from sentence_transformers import SentenceTransformer

# Load a small, fast embedding model. The FIRST run downloads it (~90 MB)
# and caches it locally; later runs are instant and fully offline.
print("Loading embedding model (first run downloads it)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Turn one piece of text into an embedding (a vector of numbers).
text = "SU01 is used to create and manage SAP user accounts."
vector = model.encode(text)

print("Success! Embedding created locally (no API, no key).")
print("Vector length (number of dimensions):", len(vector))
print("First 5 numbers:", vector[:5])
