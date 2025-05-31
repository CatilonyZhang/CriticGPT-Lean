import Mathlib
import Aesop


set_option maxHeartbeats 0


open BigOperators Real Nat Topology Rat


theorem induction_sumkexp3eqsumksq (n : ℕ) :
   (∑ k in Finset.range n, k ^ 3) = (∑ k in Finset.range n, k) ^ 2 := by
  induction n with
  | zero =>
    norm_num
  | succ n h := by
    simp_all [Finset.sum_range_succ, Nat.pow_succ, Nat.pow_zero, Nat.mul_succ, Nat.mul_zero, Nat.add_assoc]
    ring
    <;> linarith
