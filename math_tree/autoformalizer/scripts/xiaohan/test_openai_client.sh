curl https://openai.app.msh.team/v1/embeddings \
  -H "Authorization: $MSH_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["The food was delicious and the waiter..."],
    "model": "text-embedding-ada-002"
  }'
