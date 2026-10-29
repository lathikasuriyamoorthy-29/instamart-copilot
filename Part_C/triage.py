import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

ZERO_SHOT = ('You CLASSIFY customer support tickets into exactly one category: '
             'billing, delivery, technical or fraud. '
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment.
Answer: {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet balance without my permission.
Answer: {"category": "fraud", "reason": "unauthorized transaction"}
"""


def redact(text):
    text, n_email = EMAIL_RE.subn("[EMAIL]", text)
    text, n_phone = PHONE_RE.subn("[PHONE]", text)
    text, n_order = ORDER_RE.subn("[ORDER_ID]", text)
    return text, {"email": n_email, "phone": n_phone, "order_id": n_order}


def call_with_retry(messages, ticket_id, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            log.append((ticket_id, n, wait))
            print(f"retry | ticket {ticket_id} | retry number {n} | wait {wait:.4f}s")
            time.sleep(wait)


def parse_answer(content):
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        data = None

    if isinstance(data, dict) and "category" in data:
        if data["category"] in CATEGORIES:
            return "classified_json", data["category"]
        return "flagged_invalid_output", None

    found = {w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in CATEGORIES}
    if len(found) == 1:
        return "classified_keyword", found.pop()
    return "flagged_invalid_output", None


def run_pipeline(tickets, system_prompt, token_budget=100000):
    mock_llm.reset()
    total_tokens = 0
    log = []
    rows = []

    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id,
               "true_category": t.true_category, "status": None,
               "category": None, "tokens_used": 0}

        if not str(t.text).strip():
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean_text, _ = redact(t.text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean_text}]
            for m in messages:
                assert not (EMAIL_RE.search(m["content"]) or PHONE_RE.search(m["content"])
                            or ORDER_RE.search(m["content"])), "personal data in a prompt"
            try:
                response = call_with_retry(messages, t.ticket_id, log)
            except TransientLLMError:
                row["status"] = "failed_call"
            else:
                used = response.usage.total_tokens
                total_tokens += used
                row["tokens_used"] = used
                row["status"], row["category"] = parse_answer(response.choices[0].message.content)
        rows.append(row)

    return pd.DataFrame(rows), log, total_tokens