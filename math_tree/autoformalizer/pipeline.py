import re
import json

from openai import OpenAI
from autoformalizer.repl_lean_feedback.leanrepl import LeanREPL
from autoformalizer.eval_utils import lean_feedback

autoformalizer_client = OpenAI(base_url = "http://0.0.0.0:12022/v1", api_key = "casia123")
critic_client =OpenAI(base_url = "http://0.0.0.0:12023/v1", api_key = "casia123")

def generate_lean_code_from_statement(statement: str, openai_client: OpenAI) -> str:
    """
    Use OpenAI to convert a natural language mathematical statement into Lean 4 code.
    """
    prompt1 = f"""
You are an expert in formalizing mathematics using Lean4.
Your task is to accurately translate the provided mathematical statement into its corresponding Lean4 code representation.
Focus exclusively on declaring the statement within Lean4's type theory.
Your output should be a syntactically correct and well-formed Lean4 module.

Here is the mathematical statement to formalize:
{statement}

Provide ONLY the Lean4 code, without any explanations or commentary. Start with a proper module declaration and include necessary imports.
**The Lean4 code must end with `by sorry`.**
If you're unsure about any aspect, make your best attempt at formalizing the statement.
"""
    prompt = f"""
{statement}
"""     
        
    

    response = openai_client.chat.completions.create(
        model="Kimina-Autoformalizer-7B",
        messages=[{"role": "user", "content": prompt}],
    )
    
    lean_code = response.choices[0].message.content.strip()
    
    # 使用正则表达式移除注释部分
    pattern = r'/-[\s\S]*?-\/'
    lean_code = re.sub(pattern, '', lean_code, flags=re.DOTALL)
    
    return lean_code


from autoformalizer.eval_utils.lean_feedback import lean4_feedback, has_error
import json
def evluate_compile(lean_code: str, accept_sorry: bool = True):
    result = lean4_feedback(
        lean_code=lean_code,
        timeout=60,
        memory_limit=32,
        max_retries=1,
        verbose=False
    )
    
    has_any_error = has_error(result, accept_sorry)
    return result, has_any_error

def evaluate_semantics_with_openai(lean_code: str, statement: str, openai_client: OpenAI) -> dict:
    prompt_template = '''
Role: Lean & Formal Verification Expert

Input:
- Mathematical_Text: {mathematical_text_placeholder}
- Lean4Code: {lean_code_placeholder}

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
{{
  "reasons": "\\n1. Mathematical Text Analysis: [...]\n2. Lean4 Code Analysis: [...]\n3. Comparative Analysis: [...]\n4. Accuracy Confirmation: [...match confirmation or list of discrepancies...]",
  "is_assistant_correct": "Correct" or "Incorrect"
}}
'''
    # 在这里使用 .format() 方法填充内容
    prompt = prompt_template.format(
        mathematical_text_placeholder=statement,
        lean_code_placeholder=lean_code
    )

    completion = openai_client.chat.completions.create(
        model="Qwen2.5-7B-Instruct-critic_lean_mix_48k_rl",
        messages=[{"role": "user", "content": prompt}],
    )

    content = completion.choices[0].message.content.strip()
    

    return content

def extract_and_judge_correctness(feedback_string: str) -> bool:
    """
    从 OpenAI feedback 字符串中提取 'is_assistant_correct' 的值并判断。

    Args:
        feedback_string: 包含 OpenAI feedback 的字符串。

    Returns:
        如果 'is_assistant_correct' 为 'Correct'，则返回 True；否则返回 False。
    """
    # 尝试将字符串解析为 JSON 对象
    try:
        feedback_json = json.loads(feedback_string)
        correctness_value = feedback_json.get("is_assistant_correct")

        if correctness_value == "Correct":
            return True
        else:
            return False
    except json.JSONDecodeError:
        # 如果不是有效的 JSON，则尝试使用正则表达式
        # print("Warning: Feedback string is not a valid JSON. Falling back to regex.")
        # 使用正则表达式匹配 "is_assistant_correct": "Correct" 或 "Incorrect"
        # 正则表达式解释:
        # "is_assistant_correct"\s*:\s* : 匹配字面字符串 "is_assistant_correct":, 允许冒号前后有空格
        # "([Correct|Incorrect])"        : 捕获组，匹配 "Correct" 或 "Incorrect" 并捕获它
        pattern = r'"is_assistant_correct"\s*:\s*"([^"]+)"'
        match = re.search(pattern, feedback_string)

        if match:
            extracted_value = match.group(1) # 获取捕获组的内容
            if extracted_value == "Correct":
                return True
            else:
                return False
        else:
            # print("Error: Could not find 'is_assistant_correct' using regex.")
            return False # 如果正则表达式也找不到，返回 False

def formalization_pipeline(
    statement: str,
    autoformalizer_client: OpenAI,
    critic_client: OpenAI,
    max_attempts: int = 6,
    accept_sorry: bool = True
) -> dict:
    result = {
        "statement": statement,
        "lean_code": None,
        "openai_feedback": None,
        "status": "failed",
        "attempts": 0,
        "compile_result": None,
        "compile_has_any_error": None
    }
    
    current_lean_code = None
    current_openai_feedback = None

    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt

        try:
            current_lean_code = generate_lean_code_from_statement(statement, autoformalizer_client)
        except Exception as e:
            continue

        compile_result, has_any_error = evluate_compile(current_lean_code)
        result["compile_result"] = compile_result
        result["compile_has_any_error"] = has_any_error
        
        if has_any_error:
            continue

        # 3. 评估语义正确性
        # print("\nStep 3: Evaluating semantic correctness with OpenAI...")
        try:
            current_openai_feedback = evaluate_semantics_with_openai(current_lean_code, statement, critic_client)
            # print("OpenAI Feedback (raw):\n", current_openai_feedback)

            is_semantically_correct = extract_and_judge_correctness(current_openai_feedback)

            if is_semantically_correct:
                # print("\nSemantic Validity: VALID (OpenAI confirmed correctness)")
                # print("--- Formalization Pipeline Succeeded! ---")
                result["lean_code"] = current_lean_code
                try:
                    result["openai_feedback"] = json.loads(current_openai_feedback) # 成功时解析为字典
                except json.JSONDecodeError:
                    result["openai_feedback"] = current_openai_feedback # 如果解析失败，存储原始字符串
                    # print("Warning: Could not parse OpenAI feedback into JSON.")
                result["status"] = "success"
                return result # 成功时返回结果

            else:
                # print("\nSemantic Validity: INVALID (OpenAI found discrepancies)")
                # print("Retrying with a new code generation...")
                continue # 继续下一次尝试

        except Exception as e:
            # print(f"Error during semantic evaluation: {e}")
            # print("Retrying...")
            continue

    # 所有尝试都失败
    # print("\n--- All attempts failed. Formalization pipeline could not succeed. ---")
    result["status"] = "failed"
    result["lean_code"] = current_lean_code # 保留最后一次尝试的 lean_code
    try:
        result["openai_feedback"] = json.loads(current_openai_feedback) if current_openai_feedback else None
    except json.JSONDecodeError:
        result["openai_feedback"] = current_openai_feedback # 如果解析失败，存储原始字符串
        # print("Warning: Could not parse final OpenAI feedback into JSON.")
    return result

def api_call(messages):
    ret = formalization_pipeline(messages, autoformalizer_client, critic_client, max_attempts=10)
    return ret

# --- 示例用法 ---
if __name__ == "__main__":
    test_statement_1 = "Compute the conditional probability \\( A = P(X_6 > X_2 \\mid X_1 = \\max[X_1, X_2, X_3, X_4, X_5]) \\) for iid random variables \\( X_i \\). \n Prove that the answer is \\dfrac{7}{12}."
    # 并且 evaluate_semantics_with_openai 会返回 "Correct")
    print("\n--- Running pipeline for a (potentially) successful scenario ---")
    final_result_1 = formalization_pipeline(test_statement_1, autoformalizer_client, critic_client, max_attempts=3)
    print("\n--- Final Pipeline Result (Scenario 1) ---")
    print(json.dumps(final_result_1, indent=2, ensure_ascii=False))

