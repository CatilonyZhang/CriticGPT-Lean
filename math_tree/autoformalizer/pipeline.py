from openai import OpenAI

def generate_lean_code_from_statement(statement: str, openai_client: OpenAI) -> str:
    """
    Use OpenAI to convert a natural language mathematical statement into Lean 4 code.
    """
    prompt = (
        "你是一个 Lean4 代码生成器。请根据下面的数学描述，生成对应的 Lean4 定理，使用 Mathlib 和 `by` tactic 完成证明，避免使用 `sorry`。\n\n"
        "数学描述：\n"
        f"{statement}\n\n"
        "请仅返回完整的 Lean4 代码，无需其他解释。"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()

from autoformalizer.repl_lean_feedback.leanrepl import LeanREPL
from autoformalizer.eval_utils import lean_feedback

def run_lean_and_get_feedback(lean_code: str, accept_sorry: bool = True) -> dict:
    result = {
        "compiles": False,
        "errors": [],
        "context": None,
        "raw_feedback": None,
    }
    repl = LeanREPL()
    try:
        context = repl.create_context(lean_code)
        result["context"] = context
        if context is None:
            result["errors"].append("Failed to create context.")
            return result

        feedback = getattr(context, "feedback", {})
        result["raw_feedback"] = feedback

        has_err, messages = lean_feedback.has_error(
            feedback, accept_sorry=accept_sorry, return_error_messages=True
        )

        if has_err:
            result["errors"] = messages
        else:
            result["compiles"] = True
    finally:
        repl.close()

    return result

def evaluate_semantics_with_openai(lean_code: str, statement: str, openai_client: OpenAI) -> dict:
    prompt = f'''
Role: Lean & Formal Verification Expert

Input:
- Mathematical_Text: A math problem and its answer (no proof).
- Lean4Code: A Lean 4 theorem statement formalizing the problem. Proof is intentionally omitted (e.g., sorry).

Goal:
Determine if the Lean theorem statement is an exact and faithful formalization of the mathematical problem.  
**Do not evaluate or consider the answer or the proof. Your sole task is to verify the correctness of the formalization.**

Evaluation Stages (All required):

1. Mathematical Text Analysis  
   Identify all structurally and semantically relevant components of the mathematical problem, including variables, types, quantifiers, constraints, logic structure, conclusion, and so on. The analysis should be based on the actual content of the text.

2. Lean4 Code Analysis (ignore proof part)  
   Extract all structurally and semantically relevant components from the Lean statement, including variables, types, conditions, quantifiers, constraints, the final claim, and so on. The analysis should reflect the actual content present in the Lean code.

3. Comparative Analysis  
   Check for exact correspondence between the math and Lean statements; you may refer to aspects like:
   - Semantic alignment, logic structure, and quantifier correctness.
   - Preservation of constraints and boundary assumptions.
   - Accurate typing and use of variables.
   - Strict adherence to Lean's specific syntactic and semantic rules in interpreting the Lean code.
   - Syntactic validity and proper Lean usage (free from errors).
   - Use of symbols and constructs without semantic drift.
   - No missing elements, no unjustified additions, and no automatic corrections or completions.

4. Accuracy Confirmation  
   If correct: clearly confirm why all elements match.  
   If incorrect: list all mismatches and explain how each one affects correctness.

Note: While the analysis may be broad and open to interpreting all relevant features, the final judgment must be based only on what is explicitly and formally expressed in the Lean statement.  
**Do not consider or assess any part of the proof. Your judgment should be entirely about the accuracy of the statement formalization.**

Output Format:
Return exactly one JSON object:
{
  "reasons": "\n1. Mathematical Text Analysis: [...]\n2. Lean4 Code Analysis: [...]\n3. Comparative Analysis: [...]\n4. Accuracy Confirmation: [...match confirmation or list of discrepancies...]",
  "is_assistant_correct": "Correct" or "Incorrect"
}

Mathematical_Text: {statement}

Lean4Code: {lean_code}
'''

    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    content = completion.choices[0].message.content.strip()
    semantically_valid = "Correct" in content and "Incorrect" not in content

    return {
        "semantically_valid": semantically_valid,
        "openai_feedback": content,
    }

def demo_pipeline(statement: str, openai_client: OpenAI, accept_sorry: bool = True):
    print("Step 1: Generating Lean code from statement...")
    lean_code = generate_lean_code_from_statement(statement, openai_client)
    print("\nGenerated Lean code:\n", lean_code)

    print("\nStep 2: Running Lean type checker...")
    lean_feedback = run_lean_and_get_feedback(lean_code, accept_sorry=accept_sorry)
    if not lean_feedback["compiles"]:
        print("\nLean code failed to compile:")
        for err in lean_feedback["errors"]:
            print("  -", err)
        return

    print("\nStep 3: Evaluating semantic correctness with OpenAI...")
    semantic_result = evaluate_semantics_with_openai(lean_code, statement, openai_client)
    print("\nOpenAI Feedback:", semantic_result["openai_feedback"])
    print("Semantic Validity:", "valid" if semantic_result["semantically_valid"] else "invalid")

# --- Example Usage ---
if __name__ == "__main__":
    statement = "证明：对于任意实数 x，如果 x^2 - 4 = 0，则 x = 2 或 x = -2"
    client = OpenAI()
    demo_pipeline(statement, client, accept_sorry=False)


