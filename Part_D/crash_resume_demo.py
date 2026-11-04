import json
import os

import pandas as pd

import tools
from agent import STATE_DIR, SimulatedCrash, resume_agent, run_agent

RUN_ID = "S1_crash_demo"

tools.LEDGER_PATH = os.path.join("outputs", "crash_demo_ledger.jsonl")
if os.path.exists(tools.LEDGER_PATH):
    os.remove(tools.LEDGER_PATH)

scenarios = pd.read_csv("data/raw/agent_scenarios.csv", dtype=str, keep_default_na=False)
sid, cid, message, outcome, expected = scenarios.iloc[0]

try:
    run_agent(message, sid, run_id=RUN_ID, crash_after=2)
except SimulatedCrash as error:
    print("crash:", error)

with open(os.path.join(STATE_DIR, f"{RUN_ID}.json"), encoding="utf-8") as f:
    saved = json.load(f)
print("checkpoint: run id", saved["run_id"], "| scenario", saved["scenario_id"],
      "| step", saved["step"], "| messages", len(saved["messages"]),
      "| tokens used", saved["tokens_used"])
print("tools executed before the crash:", saved["tools_executed"])

result = resume_agent(RUN_ID)
executed = [s["tool"] for s in result["steps"]]
print("\nresumed run status:", result["status"], "| refund:", result["refund_inr"])
print("tools executed across both parts, in order:", executed)

assert executed == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert len(executed) == len(set(executed)), "a tool was repeated"
assert result["status"] == "REFUND" and abs(result["refund_inr"] - float(expected)) <= 0.01
print("crash and resume check passed: four executions, none repeated")