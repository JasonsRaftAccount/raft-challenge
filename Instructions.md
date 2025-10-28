# Raft AI Engineer Coding Challenge 10/28/25

### Goal: 
Prove you can build a real agent that works.

### Problem: 
Given a dummy “customer” API that returns unstructured text about orders, build an AI agent that does:
1. Takes a natural language request. Example:
“Show me all orders where the buyer was located in Ohio and total value was over 500.”
2. Calls the API to fetch raw data.
3. Uses the model to parse and structure the results.
4. Returns clean JSON output:
```
{
  "orders": [
    { "orderId": "...", "buyer": "...", "state": "OH", "total": 742.10 }
  ]
}
```
5. Must handle edge cases: context window overflow, model hallucination, unpredictable API schema changes.

### Constraints
- Use LangChain or LangGraph.
- Provide architecture diagram (simple text is fine).
- Include logging and error handling.
- Must run with one command: npm start or python main.py.

### Evaluation criteria
- Clarity of agent design.
- Correctness and determinism of output.
- Code structure and simplicity.
- How well they constrain the LLM.
  

### Starting the provided API
An API to test with is included in the `dummy_customer_api.py` file. To start it: 
```
pip install flask
python dummy_customer_api.py
```

### Testing the provided API
```
curl -X GET http://localhost:5001/api/orders
curl -X GET "http://localhost:5001/api/orders?limit=2"
curl -X GET http://localhost:5001/api/order/1003
curl -s http://localhost:5001/api/orders | jq
```