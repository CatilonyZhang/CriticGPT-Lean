import unittest

from autoformalizer.clients.lean4_client import Lean4Client
from autoformalizer.eval_utils import lean_feedback


class TestLean4ClientWithServer(unittest.TestCase):

    def setUp(self):
        """Set up the client with the actual server details."""
        self.client = Lean4Client(url="https://kimina.saas.moonshot.cn/lean4-evaluator")
        self.timeout = 60

    def test_one_pass_verify_warning(self):
        """Test a proof with warnings."""
        proof_code = """
        import Mathlib

        theorem number_theory_3784 (k m : ℕ) (hm : Odd m) (hk : 0 < k) :
            ∃ n : ℕ, 0 < n ∧ (n ^ n - m) % (2 ^ k) = 0 := by
          use 1
          constructor
          norm_num
          cases' hm with m hm
          simp [Nat.pow_succ, Nat.mul_mod, Nat.pow_mod, hm]
        """

        output = self.client.one_pass_verify(proof_code, timeout=self.timeout)
        self.assertTrue(not lean_feedback.has_error(output["response"]))
        # check warning is response
        self.assertTrue("warning" in str(output["response"]))

    def test_one_pass_verify_correct(self):
        """Test a correct proof without warnings or errors."""
        proof_code = """
        import Mathlib

        theorem algebra_260 (x : ℝ) (hx : x ≠ 0) : 1 / 2 - 1 / 3 = 1 / x ↔ x = 6 := by
          field_simp
          constructor
          intro h
          apply Eq.symm
          linarith
          intro h
          rw [h]
          norm_num
        """
        expected_output = {"error": None, "response": {"env": 0}}

        output = self.client.one_pass_verify(proof_code, timeout=self.timeout)
        self.assertEqual(output, expected_output)

    def test_one_pass_verify_error(self):
        """Test a proof that results in an error."""
        proof_code = """
        import Mathlib

        theorem algebra_158 {a b : ℕ} (ha : a > b) (h : ∀ x, x^2 - 16 * x + 60 = (x - a) * (x - b)) :
            3 * b - a = 8 := by
          have h₁ := h 0
          have h₂ := h 1
          have h₃ := h 2
          have h₄ := h 3
          simp at h₁ h₂ h₃ h₄
          ring_nf at h₁ h₂ h₃ h₄
          omega
        """

        output = self.client.one_pass_verify(proof_code, timeout=self.timeout)
        self.assertTrue(lean_feedback.has_error(output["response"]))

    def test_sorry(self):
        proof_code = """
        import Mathlib

        theorem algebra_158 {a b : ℕ} (ha : a > b) (h : ∀ x, x^2 - 16 * x + 60 = (x - a) * (x - b)) :
            3 * b - a = 8 := by sorry
        """

        output = self.client.one_pass_verify(proof_code, timeout=self.timeout)
        result = lean_feedback.parse_client_response(output)
        self.assertTrue(result["is_valid_with_sorry"])
        self.assertTrue(not result["is_valid_no_sorry"])


if __name__ == "__main__":
    unittest.main()
