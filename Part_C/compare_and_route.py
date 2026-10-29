import contextlib
import io
import os

import pandas as pd

from triage import FEW_SHOT, ZERO_SHOT, run_pipeline

TOKEN_BUDGET = 100000
DEMO_BUDGET = 6000
CLASSIFIED = ["classified_json", "classified_keyword"]
HUMAN_STATUSES = ["skipped_empty", "flagged_long", "flagged_invalid_output",
                  "budget_exhausted", "failed_call"]


def quiet_run(tickets, prompt, budget=TOKEN_BUDGET):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_pipeline(tickets, prompt, budget)


def route(row):
    if row["status"] in HUMAN_STATUSES:
        return "human_review", "NORMAL"
    if row["category"] == "fraud":
        return "security_team", "HIGH"
    if row["is_premium"]:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
os.makedirs("outputs", exist_ok=True)

# ---------- Task 6: zero-shot against few-shot ----------
summary = {}
for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
    result, log, total_tokens = quiet_run(tickets, prompt)
    done = result[result["status"].isin(CLASSIFIED)]
    correct = int((done["category"] == done["true_category"]).sum())
    summary[name] = {
        "classified": len(done),
        "correct": correct,
        "accuracy_pct": round(100 * correct / len(done), 1),
        "total_tokens": total_tokens,
        "tokens_per_classified": round(total_tokens / len(done), 1),
    }
    if name == "zero-shot":
        print("zero-shot confusion table (rows = true, columns = predicted):")
        print(pd.crosstab(done["true_category"], done["category"]))
        print()

print(pd.DataFrame.from_dict(summary, orient="index"))

# ---------- Task 7a: token budget demo ----------
demo, _, demo_tokens = quiet_run(tickets, ZERO_SHOT, DEMO_BUDGET)
demo[["ticket_id", "customer_id", "status", "category", "tokens_used"]].to_csv(
    "outputs/triage_budget_demo.csv", index=False)
print("\nbudget demo with TOKEN_BUDGET =", DEMO_BUDGET)
print("tokens used:", demo_tokens)
print("budget_exhausted tickets:", int((demo["status"] == "budget_exhausted").sum()))

# ---------- Task 7b: final run and routing ----------
final, _, final_tokens = quiet_run(tickets, FEW_SHOT, TOKEN_BUDGET)
print("\nfinal run status counts:", final["status"].value_counts().to_dict())

segments = pd.read_csv("outputs/customer_segments.csv")[
    ["customer_id", "segment_name", "is_premium"]]
final = final.merge(segments, on="customer_id", how="left")
final["is_premium"] = final["is_premium"].eq(True)

pairs = [route(r) for _, r in final.iterrows()]
final["route"] = [p[0] for p in pairs]
final["priority"] = [p[1] for p in pairs]

out = final[["ticket_id", "customer_id", "status", "category", "segment_name",
             "priority", "route", "tokens_used"]]
out.to_csv("outputs/triage_results.csv", index=False)
print("saved outputs/triage_results.csv:", out.shape)
print(out["route"].value_counts().to_dict())
print(pd.crosstab(out["route"], out["priority"]))