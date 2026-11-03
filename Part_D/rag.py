import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mock_llm

POLICY_DIR = os.path.join("data", "policies")
CHUNK_WORDS = 40

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT. Answer the question using only the supplied "
                 "context. If the context does not contain the answer, say that you could "
                 "not find the information in the policy documents.")


def load_documents(folder=POLICY_DIR):
    docs = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".txt"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                docs[name] = f.read()
    return docs


def chunk_words(text, size=CHUNK_WORDS, overlap=10):
    words = text.split()
    step = size - overlap
    chunks = []
    start = 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def build_chunks(overlap=10):
    chunks = []
    for name, text in load_documents().items():
        for piece in chunk_words(text, overlap=overlap):
            chunks.append({"source": name, "text": piece})
    return chunks


class PolicyIndex:
    def __init__(self, overlap=10):
        self.chunks = build_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"source": self.chunks[i]["source"],
                 "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in best]


DEFAULT_INDEX = PolicyIndex()


def retrieve(query, k=3, index=None):
    return (index or DEFAULT_INDEX).retrieve(query, k)


def answer(question, index=None):
    chunks = retrieve(question, 3, index)
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
    ]
    response = mock_llm.chat(messages)
    return response.choices[0].message.content, [c["source"] for c in chunks]