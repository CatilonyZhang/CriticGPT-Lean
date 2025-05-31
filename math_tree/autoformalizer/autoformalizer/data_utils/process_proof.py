import copy
import re


def remove_comments(lean_code):
    # Remove entire block comments starting with '/--' and ending with '-/' or '--/'
    lean_code = re.sub(r"/-.*?(--/|-/)", "", lean_code, flags=re.DOTALL)
    # Remove line comments with '--'
    lean_code = re.sub(r"--.*", "", lean_code)
    return lean_code


def get_statement_split_indexes(code):
    """
    This function will return the indexes of the code where the proof statement ends, like after := or := by

    One cases that this function will not work is with match statement, whihc don't ends with := or := by
    ```lean4
    import Mathlib

    theorem example : "...."
    | casea := sorry
    | caseb := sorry
    ```

    Args:
        code (str): The code that need to be splited

    Returns:
        List[int]: The indexes where the code should be splited, each for one statements in the proof

    """
    BEGIN_TOKEN = ["theorem", "lemma", "example"]
    DELAYED_TOKEN = ["let", "have"]
    ori_code = copy.copy(code)
    split_indexs = []

    # make sure the := is a seperate token
    code = remove_comments(code.replace(":=", " := "))

    # split code into tokens
    tokens = code.split()

    index = 0
    in_statement = False
    in_delayed = False
    for i, token in enumerate(tokens):
        if token in BEGIN_TOKEN:
            in_statement = True
        if token in DELAYED_TOKEN and in_statement:
            in_delayed = True
        if token in [":="]:
            while ori_code[index - len(token) : index] != token:
                index += 1
            if in_statement and not in_delayed:
                if tokens[i + 1] == "by":
                    while ori_code[index - len("by") : index] != "by":
                        index += 1
                split_indexs.append(index)
                in_statement = False
                continue
            if in_statement and in_delayed:
                in_delayed = False
                continue
        while ori_code[index - len(token) : index] != token:
            index += 1
    return split_indexs


def is_proof_splitable(proof):
    """
    This function will check if the proof is splitable or not

    Args:
        proof (str): The proof that need to be checked

    Returns:
        bool: True if the proof can be split, False otherwise
    """
    return len(get_statement_split_indexes(proof)) == 1


def split_proof(proof, mode="last"):
    """
    This function will split the proof into multiple statements

    Args:
        proof (str): The proof that need to be splited

    Returns:
        List[str]: The list of statements in the proof
    """
    split_indexs = get_statement_split_indexes(proof)
    if mode == "last":
        split_index = split_indexs[-1]
        proof_input = proof[:split_index]
        proof_output = proof[split_index:]
        # remove leading and trailing whitespaces
        if proof_output.startswith(" \n"):
            proof_input += "\n"
            proof_output = proof_output[2:]
        elif proof_output.startswith("\n"):
            proof_input += "\n"
            proof_output = proof_output[1:]
    else:
        raise ValueError(f"Invalid mode {mode}")
    return proof_input, proof_output
