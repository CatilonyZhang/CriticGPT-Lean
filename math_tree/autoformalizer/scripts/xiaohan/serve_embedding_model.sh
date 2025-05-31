# vllm serve /mnt/moonfs/kimina-m2/models/embedding/bge-m3 --served-model-name bge-m3

python -m vllm.entrypoints.openai.api_server --model /mnt/moonfs/kimina-m2/models/embedding/bge-m3 --served-model-name bge-m3 --tensor-parallel-size 8 --gpu-memory-utilization 0.95

# python -m vllm.entrypoints.openai.api_server \
#   --model /mnt/moonfs/kimina-m2/models/embedding/all-MiniLM-L6-v2  \
#   --served-model-name MiniLM-L6  --tensor-parallel-size 4 \
#   --gpu-memory-utilization 0.95 --trust-remote-code