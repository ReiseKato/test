curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "prompt": "Tell me a joke.",
    "max_tokens": 50
}'