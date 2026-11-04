import os
import shutil

import pandas as pd

from agent import RUNS_PATH, STATE_DIR, log_run, run_agent
from tools import LEDGER_PATH

for path in [RUNS_PATH, LEDGER_PATH]:
    if os.path.exists(path):
        os.remove(path)
shutil.rmtree(STATE_DIR, ignore_errors=True)

scenarios = pd.read_csv("data/raw/agent_scenarios.csv", dtype=str, keep_default_na=False)
print("columns:", scenarios.columns.tolist())

rows = []
for sid, cid, message, outcome, expected in scenarios.itertuples(index=False):
    result = run_agent(message, sid)
    log_run(result)
    if outcome.startswith("FALLBACK"):
        ok = result["status"].startswith("FALLBACK")
    else:
        ok = (result["status"] == outcome
              and abs(result["refund_inr"] - float(expected)) <= 0.01)
    rows.append({"scenario": sid, "expected": outcome, "status": result["status"],
                 "refund_inr": result["refund_inr"], "expected_refund": float(expected),
                 "model_calls": result["llm_calls"], "tokens": result["tokens_used"],
                 "match": ok})

print(pd.DataFrame(rows).to_string(index=False))

sid, cid, message, outcome, expected = scenarios.iloc[0]

demo = run_agent(message, sid, token_limit=200, run_id="S1_token_limit_demo")
print("\nS1 with token_limit=200 ->", demo["status"], "| model calls:", demo["llm_calls"],
      "| tokens used:", demo["tokens_used"], "| steps completed:", len(demo["steps"]))
print("final answer:", demo["final_answer"])

demo = run_agent(message, sid, time_limit_seconds=0, run_id="S1_timeout_demo")
print("\nS1 with time_limit_seconds=0 ->", demo["status"], "| model calls:", demo["llm_calls"],
      "| steps completed:", len(demo["steps"]))
print("final answer:", demo["final_answer"])

print("\nagent_runs.jsonl lines:", sum(1 for _ in open(RUNS_PATH, encoding="utf-8")))
print("refund_ledger.jsonl lines:", sum(1 for _ in open(LEDGER_PATH, encoding="utf-8")))