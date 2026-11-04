import re

import pandas as pd

from rag import PolicyIndex, load_documents

questions = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
docs = load_documents()

for overlap in [10, 0]:
    index = PolicyIndex(overlap=overlap)
    print(f"overlap {overlap}:")
    for q in questions.itertuples(index=False):
        if q.expected_doc == "NONE":
            continue
        sentences = re.split(r"(?<=[.!?])\s+", docs[q.expected_doc])
        fact_sentence = next((" ".join(s.split()) for s in sentences
                              if q.expected_fact.lower() in s.lower()), None)
        if fact_sentence is None:
            print("  fact not found inside one sentence ->", q.question)
            continue
        whole = any(fact_sentence in c["text"] for c in index.chunks
                    if c["source"] == q.expected_doc)
        if not whole:
            print("  sentence cut by a chunk boundary ->", q.question)
            print("    full sentence:", fact_sentence)