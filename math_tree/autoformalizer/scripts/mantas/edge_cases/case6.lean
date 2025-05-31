import Mathlib
import Aesop


set_option maxHeartbeats 0


open BigOperators Real Nat Topology Rat


theorem numbertheory_fxeq4powxp6powxp9powx_f2powmdvdf2pown (m n : ℕ) (f : ℕ → ℕ)
  (h₀ : ∀ x, f x = 4 ^ x + 6 ^ x + 9 ^ x) (h₁ : 0 < m ∧ 0 < n) (h₂ : m ≤ n) :
  f (2 ^ m) ∣ f (2 ^ n) := by
  rw [h₀, h₀]
  apply Nat.dvd_of_mod_eq_zero
  simp [Nat.add_mod, Nat.mul_mod, Nat.pow_mod, Nat.mod_mod, Nat.mod_eq_of_lt,
    show 4 % 4 = 0 by decide, show 6 % 9 = 6 by decide, show 9 % 9 = 0 by decide]
