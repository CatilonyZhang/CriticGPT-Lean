import os
from openai import OpenAI

client = OpenAI(
    api_key = "None",
    base_url = "http://116.177.238.109:8006/v1"
)

print("----- standard request -----")
prompt = r'''Complete the following Lean 4 code:

```lean4
'''

code_prefix = r'''import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
'''

stream = client.chat.completions.create(
    model = "DeepSeek-Prover-V1.5-RL",  # your model endpoint ID
    messages = [
        {"role": "user", "content": prompt + code_prefix},
    ],
    stream=True
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()

