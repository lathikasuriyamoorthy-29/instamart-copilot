import json
import os
import sqlite3
from datetime import datetime

from rag import retrieve

DB_PATH = os.path.join("data", "instamart.db")
LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")


def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1],
            "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def write_ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    line = {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool": tool, "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def issue_refund(order_id, refund_inr):
    write_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    write_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


TOOLS_SPEC = [
    {"name": "get_order",
     "description": "Look up one order in the database by its order id.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}},
    {"name": "search_policy",
     "description": "Search the policy documents and return the three best matching chunks.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "calculate_refund",
     "description": "Calculate the refund in INR from the order amount and the delivery time.",
     "parameters": {"type": "object",
                    "properties": {"order_amount": {"type": "number"},
                                   "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund",
     "description": "Issue a refund of the given amount for an order and record it in the ledger.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"},
                                   "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human",
     "description": "Send a refund above INR 1000 to a human lead and record it in the ledger.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"},
                                   "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}