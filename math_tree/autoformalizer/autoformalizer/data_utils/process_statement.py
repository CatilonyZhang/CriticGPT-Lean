def insert_informal(statement, informal):
    """
    Insert the informal problem as comment before the formal statement.
    If informal problem is found in the statement, it will return the original statement.
    WARNING: If theorem is not found in the statement, it will return the original statement.

    Args:
        statement (str): The formal statement. It is supposed to be splitable to 2 parts:
            1. Headers such as import Mathlib, and other definitions.
            2. The formal problem with sorry
        informal (str): The informal problem.
    """
    if "theorem" not in statement:
        return statement
    if informal in statement:
        return statement
    # split by \n and find the key word theorem
    lines = statement.split("\n")
    for i, line in enumerate(lines):
        if "theorem" in line:
            break

    updated = "\n".join(lines[:i] + [f"/- {informal} -/"] + lines[i:])
    return updated


if __name__ == "__main__":
    statement = """import Mathlib

theorem algebra_18785 {x : ℕ} (h : (17^6 - 17^5) / 16 = 17^x) : x = 5 := by
  sorry
"""
    print(insert_informal(statement, "Prove that $ 17^6 - 17^5 $ is divisible by 16."))
