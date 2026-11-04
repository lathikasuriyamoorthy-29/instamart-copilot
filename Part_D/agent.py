import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mock_llm
from tools import TOOL_MAP, TOOLS_SPEC

STATE_DIR = "state"
RUNS_PATH = os.path.join("outputs", "agent_runs.jsonl")

SYSTEM_PROMPT = ("You are a refund AGENT for Instamart. Use the tools to look up the order, "
                 "check the refund policy, calculate the refund, and then issue it or hand it "
                 "to a human. Finish with a short final answer that starts with REFUND, "
                 "ESCALATE, DENIED or NOT_FOUND.")


class SimulatedCrash(Exception):
    pass


def save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f"{state['run_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def log_run(result):
    os.makedirs(os.path.dirname(RUNS_PATH), exist_ok=True)
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def finish(state, status, final_answer=None):
    if final_answer is None:
        done = ", ".join(s["tool"] for s in state["steps"]) or "none"
        final_answer = (f"{status}: the run was stopped before it finished. "
                        f"Steps completed: {done}. Please hand this case to a human agent.")
    refund = 0.0
    for s in state["steps"]:
        if s["tool"] == "calculate_refund":
            refund = s["observation"]["refund_inr"]
    return {"scenario_id": state["scenario_id"], "run_id": state["run_id"],
            "status": status, "final_answer": final_answer, "steps": state["steps"],
            "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "refund_inr": refund}


def run_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after=None):
    start = time.monotonic()
    while state["llm_calls"] < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return finish(state, "FALLBACK_TIMEOUT")

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        if state["tokens_used"] > token_limit:
            return finish(state, "FALLBACK_TOKEN_LIMIT")

        message = response.choices[0].message
        if not message.tool_calls:
            answer = message.content
            return finish(state, answer.split(":")[0].strip(), answer)

        for call in message.tool_calls:
            args = call.arguments
            if isinstance(args, str):
                args = json.loads(args)
            tool = TOOL_MAP.get(call.name)
            result = tool(**args) if tool else {"error": f"unknown tool {call.name}"}

            state["messages"].append({"role": "assistant", "content": None,
                                      "tool_calls": [{"id": call.id, "name": call.name,
                                                      "arguments": args}]})
            state["messages"].append({"role": "tool", "tool_call_id": call.id,
                                      "name": call.name, "content": json.dumps(result)})
            state["step"] += 1
            state["tools_executed"].append(call.name)
            state["steps"].append({"step": state["step"], "tool": call.name,
                                   "arguments": args, "observation": result})
            save_checkpoint(state)
            if crash_after and state["step"] == crash_after:
                raise SimulatedCrash(f"simulated crash after tool execution {state['step']}")

    return finish(state, "FALLBACK_MAX_ITERATIONS")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0,
             "tools_executed": [], "steps": []}
    return run_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return run_loop(state, max_iterations, token_limit, time_limit_seconds)