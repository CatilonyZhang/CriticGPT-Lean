import re

import re
from typing import Dict, Optional, Union


def convert_braces_to_parentheses(text):
    if not text:  # Handle empty input
        return ""

    parts = text.split()
    result = []
    parentheses_content = []

    for part in parts:  # Cleaner iteration
        new_part = part.replace('{', '(').replace('}', ')')
        parentheses_content.append(new_part)

        if ')' in new_part and parentheses_content:
            result.append(' '.join(parentheses_content))
            parentheses_content.clear()

    return "\n    ".join(result)




def find_top_level_colon(s: str) -> int:
    """
    Finds the position of the first top-level colon ':' in a string.

    Args:
        s (str): The input string to search

    Returns:
        int: The index of the first top-level colon, or -1 if not found

    Notes:
        - Top-level means not nested within parentheses, braces, or brackets
        - Handles nested delimiters correctly: '({[]})'
    """
    OPENING = '({['
    CLOSING = ')}]'
    nesting_level = 0

    for i, char in enumerate(s):
        if char in OPENING:
            nesting_level += 1
        elif char in CLOSING:
            nesting_level = max(0, nesting_level - 1)  # Prevent negative nesting
        elif char == ':' and nesting_level == 0:
            return i

    return -1


def extract_theorem_lines(text: str) -> list[str]:
    """
    Extracts lines containing the theorem definition from input text.

    Args:
        text (str): The input text containing the theorem

    Returns:
        list[str]: Lines containing the theorem definition
    """
    theorem_lines = []
    lib_lines = []
    in_theorem = False

    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith('theorem'):
            in_theorem = True
        if in_theorem:
            theorem_lines.append(line)
        else:
            lib_lines.append(line)

    return lib_lines, theorem_lines


def parse_theorem_header(header: str) -> tuple[str, str]:
    """
    Splits theorem header into variables/hypotheses and conclusion.

    Args:
        header (str): The theorem header string

    Returns:
        tuple[str, str]: (variables_and_hypotheses, conclusion)

    Raises:
        ValueError: If no top-level colon is found
    """
    colon_pos = find_top_level_colon(header)
    if colon_pos == -1:
        raise ValueError("No top-level ':' found in theorem header")

    variables_and_hypotheses = header[:colon_pos].strip()
    conclusion = header[colon_pos + 1:].strip()

    return variables_and_hypotheses, conclusion



def extract_theorem_components(theorem_statement: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Parses a Lean theorem statement into its components.

    Args:
        theorem_statement (str): The complete theorem statement text

    Returns:
        Optional[Dict[str, Optional[str]]]: A dictionary containing:
            - theorem_name: Name of the theorem
            - variables_and_hypotheses: Variables and hypotheses (or None if empty)
            - conclusion: The theorem conclusion
            - proof: The proof text (or empty string if not provided)
            Returns None if parsing fails
    """
    try:
        # Extract theorem lines
        lib_lines, theorem_lines = extract_theorem_lines(theorem_statement)
        if not len(theorem_lines):
            raise ValueError("No theorem found in the provided statement")

        # Join lines and split off proof
        lib_str= '\n'.join(lib_lines)
        theorem_str = ' '.join(theorem_lines)
        header_and_conclusion, *proof_parts = theorem_str.split(':=', 1)
        proof = proof_parts[0].strip() if proof_parts else ''

        # Extract theorem name
        theorem_name_match = re.match(r'theorem\s+(\w+)', header_and_conclusion)
        if not theorem_name_match:
            raise ValueError("Could not extract theorem name")
        theorem_name = theorem_name_match.group(1)

        # Parse header
        header = header_and_conclusion[len('theorem ' + theorem_name):].strip()
        variables_and_hypotheses, conclusion = parse_theorem_header(header)

        # Normalize whitespace
        variables_and_hypotheses = ' '.join(variables_and_hypotheses.split()) or None
        conclusion = ' '.join(conclusion.split())

        return {
            'lib_str': lib_str,
            'theorem_name': theorem_name,
            'variables_and_hypotheses': variables_and_hypotheses,
            'conclusion': conclusion,
            'proof': proof
        }

    except Exception as e:
        print(f"Error parsing theorem: {str(e)}")
        return None

def negate_theorem_forall(theorem_statement):
    """
    Negates a Lean theorem by adding a '¬' in front of the entire statement, including the quantifiers.
    """
    components = extract_theorem_components(theorem_statement)
    if components is None:
        return None
    theorem_name = components['theorem_name']
    variables_and_hypotheses = components['variables_and_hypotheses']
    conclusion = components['conclusion']

    # Build the universal quantifiers
    if variables_and_hypotheses:
        quantifiers = '∀ ' + variables_and_hypotheses + ', '
    else:
        quantifiers = ''

    # Build the entire theorem body with quantifiers and conclusion
    theorem_body = quantifiers + conclusion

    # Wrap the entire theorem body in '¬ ( ... )'
    negated_theorem_name = 'neg_' + theorem_name
    lean_code = f"""theorem {negated_theorem_name} :
  ¬ ({theorem_body}) := by sorry"""
    return lean_code


def negate_theorem_partial(theorem_statement):
    """
    Negates a Lean theorem by converting universal quantifiers to existential quantifiers and adding a '¬' in front of the conclusion.
    """
    components = extract_theorem_components(theorem_statement)
    if components is None:
        return None
    theorem_name = components['theorem_name']
    variables_and_hypotheses = components['variables_and_hypotheses']
    conclusion = components['conclusion']

    # Build the existential quantifiers
    if variables_and_hypotheses:
        quantifiers = '∃ ' + variables_and_hypotheses + ', '
    else:
        quantifiers = ''

    # Add '¬' in front of the conclusion
    negated_conclusion = f'¬ ({conclusion})'

    # Build the negated theorem
    negated_theorem_name = 'neg_' + theorem_name
    lean_code = f"""theorem {negated_theorem_name} :
  {quantifiers}{negated_conclusion} := by sorry"""
    return lean_code

def negate_theorem(text):
    """Negates a theorem by applying logical negation according to standard mathematical logic rules.

    Args:
        text (str): Input text containing a theorem in Lean format.
                   Expected format: 'theorem name (params) (hyps) : conclusion := by sorry'
                   where:
                   - name: theorem identifier
                   - params: type parameters like (x y : ℝ)
                   - hyps: hypotheses like (h : x > 0)
                   - conclusion: the theorem's conclusion

    Returns:
        str: The negated theorem in Lean format.
             Returns None if input cannot be parsed as a valid theorem.

    Negation Rules:
        1. For uniqueness theorems (conclusion is an equality):
           - Original: ∀ params, hyps → (x = a)
           - Negated:  ∃ params, hyps ∧ ¬(x = a)

        2. For universal theorems (∀ quantified conclusion):
           - Original: ∀ params, hyps → P(x)
           - Negated:  ∃ params, hyps ∧ ¬P(x)

        3. For existential theorems (∃ quantified conclusion):
           - Original: ∃ params, hyps ∧ P(x)
           - Negated:  ∀ params, hyps → ¬P(x)

    Examples:
        >>> # Negating a uniqueness theorem
        >>> negate_theorem("theorem algebra_5 (x y : ℝ) (h : x^2 - 6*x + y^2 + 2*y = 9) : (x, y) = (3, -1) := by sorry")
        'theorem negated_algebra_5 : ∃ (x y : ℝ) (h : x^2 - 6*x + y^2 + 2*y = 9), ¬((x, y) = (3, -1)) := by sorry'

        >>> # Negating a universal theorem
        >>> negate_theorem("theorem all_positive (n : ℕ) : n > 0 := by sorry")
        'theorem negated_all_positive : ∃ (n : ℕ), ¬(n > 0) := by sorry'
    """
    components = extract_theorem_components(text)
    if components is None:
        return None
    theorem_name = components['theorem_name']
    variables_and_hypotheses = components['variables_and_hypotheses']
    conclusion = components['conclusion']

    if variables_and_hypotheses:
        quantifiers = '∃ ' + convert_braces_to_parentheses(variables_and_hypotheses) + ', '
    else:
        quantifiers = ''

    # Add '¬' in front of the conclusion
    negated_conclusion = f'¬ ({conclusion})'

    # Build the negated theorem
    negated_theorem_name = 'negated_' + theorem_name
    lean_code = f"""theorem {negated_theorem_name} :
    {quantifiers}{negated_conclusion} := by sorry"""

    return lean_code




if __name__ == "__main__":
    # Unit tests with the sample theorems
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
        """
        theorem number_theory_73 {a m n : ℕ} (ha : 1 < a)
        (hm : 0 < m) (hn : 0 < n) :
        Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 := by
        sorry
        """,
        """
        theorem number_theory_83
        (P : ℕ → Prop)
        (hP : P = fun n ↦ (2 * n)! < 2^(2 * n) * (n!)^2) :
        ∀ n, 0 < n → P n := by
        sorry
        """,
        """
        theorem number_theory_211 {p : ℕ} (hp : Nat.Prime p) (a : ZMod p) :
        a * a = 1 ↔ a = 1 ∨ a = -1 :=
        sorry
        """,
        """
        theorem number_theory_365 : (3^100000) % 35 = 1 := by sorry
        """,
        """
        theorem number_theory_60 :
        ∀ n : ℕ, 6 < n → ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a + b = n ∧ Int.gcd a b = 1 := by
        sorry
        """,
        """
        theorem number_theory_63
        (f : ℕ → ℤ)
        (h₀ : f 1 = 1)
        (h₁ : f 2 = 5)
        (h₂ : ∀ n, f (n + 2) = f (n + 1) + 2 * f n) :
        ∀ n, f n = 2^n + (-1)^n := by
        sorry
        """,
        """
        theorem exists_positive_square : 
        ∃ (x : ℝ), x > 0 ∧ x^2 = 4
        """,
        """
        theorem algebra_5 (x y : ℝ) (h : x^2 - 6 * x + y^2 + 2 * y = 9) :
    (x, y) = (3, -1) := by sorry
        """
    ]

    # # Test negate_theorem
    # print("Testing negate_theorem:")
    # for idx, theorem in enumerate(test_theorems):
    #     print(f"Test {idx+1}")
    #     negated = negate_theorem_forall(theorem)
    #     if negated:
    #         print(negated)
    #         print("\n" + "-"*50 + "\n")

    # Test negate_theorem2
    print("Testing negate_theorem2:")
    for idx, theorem in enumerate(test_theorems):
        print(f"Test {idx+1}")
        negated = negate_theorem(theorem)
        if negated:
            print(negated)
            print("\n" + "-"*50 + "\n")

