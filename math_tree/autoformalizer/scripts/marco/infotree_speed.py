import uuid
from time import time

from autoformalizer.clients.lean4_client import Lean4Client

if __name__ == "__main__":
    lean_file = "scripts/marco/test.lean"

    lean4_client = Lean4Client(
        url="https://kimina.saas.moonshot.cn/lean4-evaluator",
    )
    with open(lean_file, "r") as f:
        lean_code = f.read()
    codes = [{"code": lean_code, "custom_id": str(uuid.uuid4())}]

    start = time()
    response = lean4_client.one_pass_verify_batch(
        codes=codes, timeout=10, infotree_type=None
    )
    time_base = time() - start

    start = time()
    response = lean4_client.one_pass_verify_batch(
        codes=codes, timeout=10, infotree_type="full"
    )
    time_full = time() - start

    start = time()
    response = lean4_client.one_pass_verify_batch(
        codes=codes, timeout=10, infotree_type="tactics"
    )
    time_tactics = time() - start

    start = time()
    response = lean4_client.one_pass_verify_batch(
        codes=codes, timeout=10, infotree_type="original"
    )
    time_original = time() - start

    print(f"Time for `full` mode: {time_full - time_base:.3f} seconds")
    print(f"Time for `tactics` mode: {time_tactics - time_base:.3f} seconds")
    print(f"Time for `original` mode: {time_original - time_base:.3f} seconds")
