import Mathlib
import Aesop


set_option maxHeartbeats 0


open BigOperators Real Nat Topology Rat


theorem mathd_algebra_208 : Real.sqrt 1000000 - 1000000 ^ ((1 : ℝ) / 3) = 900 := by
  rw [sub_eq_add_neg]
  norm_num [Real.sqrt_eq_rpow, Real.rpow_def_of_pos, show (0 : ℝ) < 3 by norm_num]
