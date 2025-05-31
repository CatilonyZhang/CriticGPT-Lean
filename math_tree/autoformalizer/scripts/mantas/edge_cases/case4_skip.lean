import Mathlib
import Aesop


set_option maxHeartbeats 0


open BigOperators Real Nat Topology Rat


theorem induction_sumkexp3eqsumksq (n : ℕ) :
  (∑ k in Finset.range n, k ^ 3) = (∑ k in Finset.range n, k) ^ 2 := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp_all only [Finset.sum_range_succ, Nat.pow_succ, Nat.mul_succ]
    ring_nf
    sorry

  skip