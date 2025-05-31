import unittest
import re
import random
import string
import pandas as pd
from autoformalizer.model_utils.infer_hf_dataset import negate_theorem


class TestTheoremNegation(unittest.TestCase):
    """Test suite for the theorem negation functionality."""
    
    def setUp(self):
        """Set up test cases for theorem negation."""
        self.test_cases = {
            'simple': '''abbrev sol := sorry
theorem solution : sol % 1000 = 50 := by sorry''',
            
            'multiline': '''abbrev solution : ℕ := sorry
theorem is_solution (n : ℕ) :
    n = solution ↔
    ∃ x : Fin 4 → ℕ,
      (∀ i j, i < j → x i < x j) ∧
      (∀ i, x i ^ 2 + ∑ j : Fin 4, x j ^ 2 = n) := by sorry''',
            
            'multiple_theorems': '''theorem first_thm : 2 + 2 = 4 := by sorry
theorem second_thm : 3 * 3 = 9 := by sorry''',
            
            'complex_math': '''theorem complex_math : ∀ x y : ℝ, x < y → ∃ z, x < z ∧ z < y := by sorry''',
            
            'with_abbrev': '''abbrev A := Type
abbrev B := Type
theorem type_theorem : A → B := by sorry'''
        }
    def test_simple_theorem_negation(self):
        """Test negation of a simple single-line theorem."""
        original = self.test_cases['simple']
        result = negate_theorem(original)
        
        print("\n=== Test Simple Theorem Negation ===")
        print("\nOriginal:")
        print(original)
        print("\nNegated:")
        print(result)
        print("\n" + "="*50)
        
        self.assertIn('negated_solution', result)
        self.assertIn('¬(sol % 1000 = 50)', result)
        self.assertIn(':= by sorry', result)

    def test_multiline_theorem_negation(self):
        """Test negation of a complex multiline theorem with type parameters."""
        original = self.test_cases['multiline']
        result = negate_theorem(original)
        
        print("\n=== Test Multiline Theorem Negation ===")
        print("\nOriginal:")
        print(original)
        print("\nNegated:")
        print(result)
        print("\n" + "="*50)
        
        self.assertIn('negated_is_solution', result)
        self.assertIn('(n : ℕ)', result)
        self.assertTrue(result.count('theorem') == 1)
        self.assertIn('¬(', result)
        self.assertIn(':= by sorry', result)

    def test_multiple_theorems(self):
        """Test that only the last theorem is negated when multiple theorems exist."""
        original = self.test_cases['multiple_theorems']
        result = negate_theorem(original)
        
        print("\n=== Test Multiple Theorems ===")
        print("\nOriginal:")
        print(original)
        print("\nNegated:")
        print(result)
        print("\n" + "="*50)
        
        self.assertIn('first_thm : 2 + 2 = 4', result)
        self.assertIn('negated_second_thm', result)
        self.assertIn('¬(3 * 3 = 9)', result)

    def test_complex_mathematical_notation(self):
        """Test negation of theorem with complex mathematical notation."""
        original = self.test_cases['complex_math']
        result = negate_theorem(original)
        
        print("\n=== Test Complex Mathematical Notation ===")
        print("\nOriginal:")
        print(original)
        print("\nNegated:")
        print(result)
        print("\n" + "="*50)
        
        self.assertIn('negated_complex_math', result)
        self.assertIn('¬(∀ x y : ℝ, x < y → ∃ z, x < z ∧ z < y)', result)

    def test_theorem_with_abbreviations(self):
        """Test negation of theorem with preceding abbreviations and comments."""
        original = self.test_cases['with_abbrev']
        result = negate_theorem(original)
        
        print("\n=== Test Theorem with Abbreviations ===")
        print("\nOriginal:")
        print(original)
        print("\nNegated:")
        print(result)
        print("\n" + "="*50)
        
        self.assertIn('abbrev A := Type', result)
        self.assertIn('abbrev B := Type', result)
        self.assertIn('negated_type_theorem', result)
        self.assertIn('¬(A → B)', result)

    def test_no_theorem_present(self):
        """Test behavior when no theorem is present in the text."""
        original = "abbrev solution := 42"
        result = negate_theorem(original)
        
        print("\n=== Test No Theorem Present ===")
        print("\nOriginal:")
        print(original)
        print("\nNegated:")
        print(result)
        print("\n" + "="*50)
        
        self.assertEqual(result, original)

    def test_random_name_generation(self):
        """Test that random names are generated when theorem name is missing."""
        original = "theorem : P ∧ Q := by sorry"
        result = negate_theorem(original)
        
        print("\n=== Test Random Name Generation ===")
        print("\nOriginal:")
        print(original)
        print("\nNegated:")
        print(result)
        print("\n" + "="*50)
        
        self.assertRegex(result, r'theorem negated_theorem_[a-z]{6}_[0-9]{4}')
        self.assertIn('¬(P ∧ Q)', result)

if __name__ == "__main__":
    """It can be run directly using:
        
    python3 -m autoformalizer.bootstrap.tests.test_negated_match
    
    
    Test runner for theorem negation functionality.

    This script provides a test suite for the theorem negation function, which transforms
    mathematical theorems by adding logical negation.

    Test Cases:
    -----------
    The test cases are defined in the setUp method of TestTheoremNegation class.
    You can modify existing test cases or add new ones by updating self.test_cases dict:

    Current test categories:
    - 'simple': Basic single-line theorem
    - 'multiline': Complex theorem with type parameters and multiple lines
    - 'multiple_theorems': Multiple theorems in sequence
    - 'complex_math': Theorem with complex mathematical notation
    - 'with_abbrev': Theorem with abbreviations and comments

    Adding New Test Cases:
    --------------------
    To add a new test case:
    1. Add it to self.test_cases in setUp():
        self.test_cases['new_case'] = '''
        theorem new_theorem : your_theorem_statement := by sorry
        '''

    2. Create a corresponding test method:
        def test_new_feature(self):
            '''Test description'''
            original = self.test_cases['new_case']
            result = negate_theorem(original)
            print("\n=== Test New Feature ===")
            print("\nOriginal:")
            print(original)
            print("\nNegated:")
            print(result)
            print("\n" + "="*50)
            self.assertIn('expected_content', result)

    Output Format:
    -------------
    For each test case, the output shows:
    1. Test case name
    2. Original theorem text
    3. Negated theorem text
    4. Visual separator

    Example:
    --------
    === Test Simple Theorem Negation ===
    Original:
    theorem solution : sol % 1000 = 50 := by sorry
    
    Negated:
    theorem negated_solution : ¬(sol % 1000 = 50) := by sorry
    ==================================================
    """
    unittest.main()