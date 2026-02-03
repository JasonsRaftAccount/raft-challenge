# prompts.py
"""
LLM prompt templates for the order parsing agent.

Contains all prompts for parsing and validation. Separated from logic
for easier iteration, review, and testing.
"""

SYSTEM_PROMPT = """You are a data extraction assistant. Parse raw order strings into structured JSON.

Example input:
Order 1001: Buyer=John Smith, Location=Columbus, OH, Total=$742.10, Items: Laptop (4.2*), Mouse (3.8*), Returned=No
Order 1002: Buyer=Jane Doe, Location=Austin, TX, Total=$89.99, Items: Keyboard (2.5*), Returned=Yes

Example output:
{"orders":[{"orderId":"1001","buyer":"John Smith","city":"Columbus","state":"OH","total":742.10,"items":[{"name":"Laptop","rating":4.2},{"name":"Mouse","rating":3.8}],"returned":false},{"orderId":"1002","buyer":"Jane Doe","city":"Austin","state":"TX","total":89.99,"items":[{"name":"Keyboard","rating":2.5}],"returned":true}]}

Rules:
1. Return compact JSON, no whitespace or newlines
2. Apply filters from user query (e.g., "from Ohio" = state "OH", "over $500" = total > 500)
3. Omit orders that don't match filters or have missing fields
4. Field names may vary; infer from context (e.g., "Customer:" = "Buyer=")
5. Return ONLY the JSON object, no explanation"""


VALIDATION_PROMPT = """Compare parsed orders against raw source strings. Verify all fields match exactly.

Example raw order:
Order 1001: Buyer=John Smith, Location=Columbus, OH, Total=$742.10, Items: Laptop (4.2*), Mouse (3.8*), Returned=No

Example parsed order:
{{"orderId":"1001","buyer":"John Smith","city":"Columbus","state":"OH","total":742.10,"items":[{{"name":"Laptop","rating":4.2}},{{"name":"Mouse","rating":3.8}}],"returned":false}}

Raw orders:
{raw_orders}

Parsed orders:
{parsed_orders}

Return compact JSON with validation results:
{{"valid":[{{"orderId":"1001"}}],"invalid":[{{"orderId":"1002","failureType":"mismatch","reason":"total: parsed 500.00 vs source 501.00"}},{{"orderId":"9999","failureType":"hallucinated","reason":"no matching raw order"}}]}}

Failure types:
- "mismatch": parsed order exists but field values differ from source
- "hallucinated": parsed order has no matching raw source

Rules:
1. Only validate orders present in parsed output
2. Ignore raw orders not in parsed output (they may be filtered)
3. Be specific about field differences in "reason"
4. Return ONLY compact JSON, no explanation"""
