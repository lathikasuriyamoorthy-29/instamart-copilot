from collections import Counter

import pandas as pd

from rag import PolicyIndex, build_chunks, retrieve

chunks = build_chunks()
print("chunks:", len(chunks), dict(Counter(c["source"] for c in chunks)))
words_first = chunks[0]["text"].split()
words_second = chunks[1]["text"].split()
assert len(words_first) == 40 and words_second[:10] == words_first[30:40], "overlap is wrong"
print("chunk 1 starts with the last 10 words of chunk 0: True")

print("\ntop 3 for 'refund policy late delivery':")
for r in retrieve("refund policy late delivery"):
    print(" ", r["source"], r["score"])

questions = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)


def evaluate(index):
    from rag import answer
    stats = {"answerable": 0, "doc_in_top3": 0, "fact_in_answer": 0,
             "unanswerable": 0, "fallback_given": 0}
    failures = []
    for q in questions.itertuples(index=False):
        text, sources = answer(q.question, index)
        if q.expected_doc == "NONE":
            stats["unanswerable"] += 1
            if text.startswith("I could not find"):
                stats["fallback_given"] += 1
            else:
                failures.append((q.question, text))
        else:
            stats["answerable"] += 1
            doc_ok = q.expected_doc in sources
            fact_ok = q.expected_fact.lower() in text.lower()
            stats["doc_in_top3"] += doc_ok
            stats["fact_in_answer"] += fact_ok
            if not (doc_ok and fact_ok):
                failures.append((q.question, text))
    return stats, failures


for label, overlap in [("overlap 10", 10), ("overlap 0", 0)]:
    index = PolicyIndex(overlap=overlap)
    stats, failures = evaluate(index)
    print(f"\n=== {label}: {len(index.chunks)} chunks ===")
    print(f"answerable: expected doc in top 3 = {stats['doc_in_top3']}/{stats['answerable']}, "
          f"expected fact in answer = {stats['fact_in_answer']}/{stats['answerable']}")
    print(f"unanswerable: fallback given = {stats['fallback_given']}/{stats['unanswerable']}")
    for question, text in failures:
        print("FAILED:", question, "->", text)