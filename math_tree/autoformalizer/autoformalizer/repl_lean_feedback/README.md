# Lean4 feedback for prover model

This code enables interaction with Lean4 from Python, allowing to apply tactics from Python with real-time feedback from Lean4.

The code contains 5 main files:
- `leanrepl.py`: defines the LeanREPL class, which allows to start a LeanREPL process, send commands, get feedback, and create new states.
- `state.py`: defines the State class, which corresponds to a Lean4 state (can be seen as a node in a proof tree) carrying the following information:
    - a parent state,
    - a list of children states,
    - the theorem statement of the initial state,
    - the list of previous tactics applied to the initial state to reach the current state,
    - the list of current goals,
    - the list of current messages (info, warnings, errors).
- `context.py`: defines the Context class, which corresponds to everything that would be written in the same Lean4 file, above the statement that we want to prove (imported libraries, definitions, lemmas, previous theorems, etc.).
- `bfs.py`: implements a naive best-first search with constant value function and random policy over a set of finite tactics.
- `demo.py` (in `scripts/marco/`): shows how to use the code with a simple example, by:
    - creating a new LeanREPL object,
    - creating a new Context object within the LeanREPL object,
    - initializing a new theorem to prove within the Context object,
    - applying tactics one by one, getting Lean4 feedback each time,
    - verifying that we completed the proof,
    - printing the whole proof sequence.

### Sample output from `main.py`:
```
New LeanREPL created.
---
New Context created with:
import Mathlib
---
New theorem initialized with:
theorem algebra_9472 (x : ℝ) (hx : x^2 - 6 * x + 5 = 0) : x = 1 ∨ x = 5
---
Current goals: ['x : ℝ\nhx : x ^ 2 - 6 * x + 5 = 0\n⊢ x = 1 ∨ x = 5']
Tactic: have: x ^ 2 - 6 * x + 5 = (x-5)*(x-1) := by ring
New goals: ['x : ℝ\nhx : x ^ 2 - 6 * x + 5 = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\n⊢ x = 1 ∨ x = 5']
---
Current goals: ['x : ℝ\nhx : x ^ 2 - 6 * x + 5 = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\n⊢ x = 1 ∨ x = 5']
Tactic: rw [this] at hx
New goals: ['x : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\n⊢ x = 1 ∨ x = 5']
---
Current goals: ['x : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\n⊢ x = 1 ∨ x = 5']
Tactic: have eq': x-5 = 0 ∨ x - 1 = 0 := mul_eq_zero.mp hx
New goals: ["x : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\neq' : x - 5 = 0 ∨ x - 1 = 0\n⊢ x = 1 ∨ x = 5"]
---
Current goals: ["x : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\neq' : x - 5 = 0 ∨ x - 1 = 0\n⊢ x = 1 ∨ x = 5"]
Tactic: rcases eq' with e5 | e1
New goals: ['case inl\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne5 : x - 5 = 0\n⊢ x = 1 ∨ x = 5', 'case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
---
Current goals: ['case inl\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne5 : x - 5 = 0\n⊢ x = 1 ∨ x = 5', 'case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
Tactic: right
New goals: ['case inl.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne5 : x - 5 = 0\n⊢ x = 5', 'case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
---
Current goals: ['case inl.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne5 : x - 5 = 0\n⊢ x = 5', 'case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
Tactic: rw [← add_zero 5, ← e5]
New goals: ['case inl.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne5 : x - 5 = 0\n⊢ x = 5 + (x - 5)', 'case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
---
Current goals: ['case inl.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne5 : x - 5 = 0\n⊢ x = 5 + (x - 5)', 'case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
Tactic: simp
New goals: ['case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
---
Current goals: ['case inr\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 ∨ x = 5']
Tactic: left
New goals: ['case inr.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1']
---
Current goals: ['case inr.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1']
Tactic: rw [← add_zero 1, ← e1]
New goals: ['case inr.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 + (x - 1)']
---
Current goals: ['case inr.h\nx : ℝ\nhx : (x - 5) * (x - 1) = 0\nthis : x ^ 2 - 6 * x + 5 = (x - 5) * (x - 1)\ne1 : x - 1 = 0\n⊢ x = 1 + (x - 1)']
Tactic: simp
New goals: []
---
Proof is complete!
---
Average time per tactic: 0.008 seconds
Total time for the proof: 0.085 seconds
```