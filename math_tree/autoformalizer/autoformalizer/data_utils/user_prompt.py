

def get_user_prompt(natural_language: str, 
                    has_header: bool, 
                    theorem_names: list, 
                    source: str, 
                    include_source: bool):
    
    if include_source:
        user_prompt = f"Formalize the following natural language mathematics statement from {source} in Lean 4 "
    else:
        user_prompt = "Formalize the following natural language mathematics statement in Lean 4 "
    user_prompt += "with header of import and open statements. " if has_header else "without a header. "
    user_prompt += "Only use the following strings to name the theorems: " + ", ".join(theorem_names) + ".\n\n"
    user_prompt += natural_language

    return user_prompt