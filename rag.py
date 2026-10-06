import os
import json

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Loaded once when this module is imported, then reused for every embed call.
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents(folder="data"):
    """Load every .txt and .pdf file in `folder`.

    Returns a list of dicts, one per document:
        {"source": <filename>, "text": <full text of the document>}
    """
    documents = []

    # os.listdir gives the file names inside the folder. sorted() = stable order.
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)

        if filename.endswith(".txt"):
            # Plain text: just read the whole file.
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        elif filename.endswith(".pdf"):
            # PDF: extract text page by page using pypdf.
            reader = PdfReader(path)
            pages = [page.extract_text() for page in reader.pages]
            text = "\n".join(pages)

        else:
            # Skip anything that isn't a .txt or .pdf (e.g. .DS_Store).
            continue

        documents.append({"source": filename, "text": text})

    return documents


def chunk_text(text, chunk_size=120, overlap=20):
    """Split one document's text into overlapping chunks of words.

    chunk_size = how many words per chunk.
    overlap    = how many words each chunk shares with the previous one.
    Returns a list of chunk strings.
    """
    words = text.split()  # split on whitespace -> list of words

    # Guard: the step must be positive or we'd loop forever.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap  # how far we advance each time
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]        # a window of words
        chunks.append(" ".join(chunk_words))  # join back into a string
        start += step                          # slide the window forward

    return chunks


def build_chunks(documents, chunk_size=120, overlap=20):
    """Chunk every document, keeping each chunk tied to its source file.

    Returns a list of dicts: {"text": <chunk>, "source": <filename>}
    """
    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"], chunk_size, overlap):
            all_chunks.append({"text": chunk, "source": doc["source"]})
    return all_chunks


def embed_chunks(chunks):
    """Add an 'embedding' vector to every chunk dict, in place.

    Encodes all chunk texts in ONE batch call (faster than one at a time),
    then converts each NumPy vector to a plain Python list so it can be
    saved as JSON later.
    """
    texts = [chunk["text"] for chunk in chunks]
    vectors = EMBEDDING_MODEL.encode(texts)  # shape: (num_chunks, 384)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()  # NumPy array -> plain list

    return chunks


def save_index(chunks, path="index.json"):
    """Save chunks (with their embeddings) to a JSON file on disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {path}")


def load_index(path="index.json"):
    """Load previously saved chunks (with embeddings) back from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a, b):
    """How closely two vectors point in the same direction (-1 to 1)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(question, chunks, top_k=3):
    """Return the top_k chunks whose meaning is closest to `question`.

    Each result is a dict: {"score": ..., "text": ..., "source": ...}
    """
    question_vector = EMBEDDING_MODEL.encode(question)

    results = []
    for chunk in chunks:
        score = cosine_similarity(question_vector, chunk["embedding"])
        results.append({
            "score": score,
            "text": chunk["text"],
            "source": chunk["source"],
        })

    # Highest similarity first.
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def build_prompt(question, results):
    """Combine retrieved chunks + the question into one prompt for Claude."""
    context_blocks = []
    for r in results:
        context_blocks.append(f"[Source: {r['source']}]\n{r['text']}")
    context = "\n\n".join(context_blocks)

    return (
        "Answer the question using ONLY the context below. "
        "Cite the source filename for any claim you make. "
        "If the context doesn't contain the answer, say so — don't guess.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

# Quick manual test: run `python rag.py` to build the index AND try a search.
if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.\n")

    chunks = build_chunks(docs, chunk_size=120, overlap=20)
    print(f"Split into {len(chunks)} chunks total.\n")

    chunks = embed_chunks(chunks)
    print(f"Embedded all {len(chunks)} chunks.")
    print(f"Each embedding has {len(chunks[0]['embedding'])} numbers.\n")

    save_index(chunks, "index.json")

    # Try a real search against the index we just built.
    question = "How quickly must access be removed after someone leaves the company?"
    print(f"\nSearching for: {question}\n")
    top_results = search(question, chunks, top_k=3)
    for r in top_results:
        print(f"  score={r['score']:.3f}  source={r['source']}")
        print(f"    {r['text'][:150]}...\n")

    prompt = build_prompt(question, top_results)
    print("\n----- PROMPT SENT TO CLAUDE -----")
    print(prompt)
    print("----- END PROMPT -----")