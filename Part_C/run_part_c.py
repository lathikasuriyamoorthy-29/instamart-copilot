import pandas as pd

from triage import (BASE_DELAY, EMAIL_RE, FEW_SHOT, MAX_INPUT_TOKENS, ORDER_RE,
                    PHONE_RE, ZERO_SHOT, est_tokens, redact, run_pipeline)

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
print("tickets:", len(tickets))

# Task 1: guardrails
tokens = tickets["text"].map(est_tokens)
empty = tickets["text"].str.strip() == ""
long_ = tokens > MAX_INPUT_TOKENS
print("empty tickets:", int(empty.sum()), "| long tickets:", int(long_.sum()))
print("token estimates of the long tickets:", tokens[long_].tolist())

# Task 2: redaction over all 164 tickets
counts = {"email": 0, "phone": 0, "order_id": 0}
left = 0
for text in tickets["text"]:
    clean, n = redact(text)
    for key in counts:
        counts[key] += n[key]
    left += (len(EMAIL_RE.findall(clean)) + len(PHONE_RE.findall(clean))
             + len(ORDER_RE.findall(clean)))
print("substitutions:", counts, "| total:", sum(counts.values()), "| left afterwards:", left)

# Task 3: prompt rules
assert "CLASSIFY" in ZERO_SHOT and "CLASSIFY" in FEW_SHOT
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT
for category in ["billing", "delivery", "technical", "fraud"]:
    assert category in ZERO_SHOT and category in FEW_SHOT
print("prompt checks passed | prompt tokens: zero-shot", est_tokens(ZERO_SHOT),
      "| few-shot", est_tokens(FEW_SHOT))

# Tasks 4 and 5: one full zero-shot run
result, log, total_tokens = run_pipeline(tickets, ZERO_SHOT)
print("status counts:", result["status"].value_counts().to_dict())
print("retries:", len(log), "| failed calls:", int((result["status"] == "failed_call").sum()))
in_range = all(BASE_DELAY * 2 ** n * 0.9 <= w <= BASE_DELAY * 2 ** n * 1.15 for _, n, w in log)
print("every wait inside its jitter range:", in_range)
print("total tokens:", total_tokens)