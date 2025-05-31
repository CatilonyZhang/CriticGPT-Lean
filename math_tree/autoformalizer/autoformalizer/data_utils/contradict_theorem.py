from autoformalizer.data_utils.negate_theorem import extract_theorem_components


def contradict_theorem(text):
    """Generates a Lean theorem to check if a theorem's hypotheses are self-contradictory.

    Args:
        text (str): Input text containing a theorem in Lean format.
               Expected format: 'theorem name (params) (hyps) : conclusion := by sorry'
               where:
               - name: theorem identifier
               - params: type parameters like (x y : ℝ)
               - hyps: hypotheses like (h : x > 0)
               - conclusion: the theorem's conclusion

    Returns:
        str: A Lean theorem that attempts to derive False from just the hypotheses.
             Returns None if input cannot be parsed as a valid theorem.

    The function works by:
        1. Extracting the hypotheses and parameters from the input theorem
        2. Creating a new theorem that keeps the same hypotheses but tries to prove False
        3. If the resulting theorem can be proved, the original hypotheses are contradictory

    Examples:
        >>> # Checking consistency of hypotheses that are contradictory
        >>> contradict_theorem('''
        ... theorem impossible (x : ℝ)
        ... (h1 : x > 0) (h2 : x < 0) :
        ... x = 1 := by sorry
        ... ''')
        '''
        theorem impossible_contradiction (x : ℝ)
        (h1 : x > 0) (h2 : x < 0) :
        False := by sorry
        '''

        >>> # Checking consistency of valid hypotheses
        >>> contradict_theorem('''
        ... theorem valid (n : ℕ)
        ... (h : n > 0) :
        ... n ≥ 0 := by sorry
        ... ''')
        '''
        theorem valid_contradiction (n : ℕ)
        (h : n > 0) :
        False := by sorry
        '''
    """

    components = extract_theorem_components(text)
    if components is None:
        return None
    lib_name = components["lib_str"]
    theorem_name = components["theorem_name"]
    variables_and_hypotheses = components["variables_and_hypotheses"]

    # Build the contradiction theorem
    lean_code = f"""{lib_name}
theorem contradicted_{theorem_name} {variables_and_hypotheses} :
    False := by sorry"""

    return lean_code


if __name__ == "__main__":
    # Unit tests with sample theorems
    test_theorems = [
        """
theorem number_theory_3
        (n : ℕ)
        (h₀ : n > 0) :
        (∑ j in Finset.Icc 1 n, j) = n * (n + 1) / 2 ∧
        (∑ j in Finset.Icc 1 n, j^2) = n * (n + 1) * (2 * n + 1) / 6 ∧
        (∑ j in Finset.Icc 1 n, j^3) = (n * (n + 1) / 2)^2 := by
        sorry
        """,
        """
theorem number_theory_42
        (g : ℕ → ℕ)
        (h₀ : g 1 = 2)
        (h₁ : ∀ n, g (n + 1) = 2^(g n)) :
        g 4 = 2^(2^(2^2)) := by
        sorry
        """,
        """
theorem number_theory_59
        (k : ℕ)
        (h₀ : 0 < k) :
        Int.gcd (3 * k + 2) (5 * k + 3) = 1 := by
        sorry
        """,
    ]

    print("Testing contradict_theorem:")
    for idx, theorem in enumerate(test_theorems):
        print(f"\nTest {idx+1}:")
        contradiction = contradict_theorem(theorem)
        print("original theorem:")
        print(theorem)
        print("Generated contradiction theorem:")
        print(contradiction)
        print("\nExpected result:")
        print(
            "This theorem's hypotheses should be consistent (contradiction unprovable)"
        )
