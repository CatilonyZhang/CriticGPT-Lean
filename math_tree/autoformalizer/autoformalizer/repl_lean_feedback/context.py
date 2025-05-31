from autoformalizer.repl_lean_feedback.repl_utils import (
    parse_error_message,
    parse_messages,
)


class Context:
    def __init__(self, id, code=None, is_valid=False, messages=None, response=None):
        self.id = id  # Identifier (int)
        self.code = (
            code or ""
        )  # String of Lean4 code containing imports, definitions, lemmas, etc. (str)
        self.states = (
            {}
        )  # Mapping from context id to State objects that live in this context, for debugging
        self.is_valid = is_valid
        self.messages = messages
        self.response = response

    @classmethod
    def from_response(cls, response, code):
        error_msg = response.get("error", None)
        if error_msg is not None:
            is_valid = False
            messages = {"severity": "error", "message": error_msg}
            return cls(
                id=None,
                code=code,
                is_valid=is_valid,
                messages=[messages],
                response=response,
            )

        lean_response = response.get("response", None)
        if lean_response is None:
            is_valid = False
            messages = {"severity": "error", "message": "No response from Lean4"}
            return cls(
                id=None,
                code=code,
                is_valid=is_valid,
                messages=[messages],
                response=response,
            )

        if "messages" in lean_response:
            is_valid = True
            messages = parse_messages(lean_response["messages"])
            if any([msg["severity"] == "error" for msg in messages]):
                is_valid = False
            context_id = lean_response.get("env", None)
            return cls(
                id=context_id,
                code=code,
                is_valid=is_valid,
                messages=messages,
                response=response,
            )

        if "message" in lean_response:
            is_valid = False
            messages = parse_error_message([lean_response])
            context_id = lean_response.get("env", None)
            return cls(
                id=context_id,
                code=code,
                is_valid=is_valid,
                messages=messages,
                response=response,
            )

        if "env" in lean_response:
            # no messages and no error message
            is_valid = True
            context_id = lean_response["env"]
            return cls(
                id=context_id,
                code=code,
                is_valid=is_valid,
                messages=[],
                response=response,
            )

        return cls(
            id=None,
            code=code,
            is_valid=False,
            messages=[
                {
                    "severity": "error",
                    "message": "Unhandled response from context.from_response",
                }
            ],
            response=response,
        )

    def add_code(self, new_code):
        """
        Append new code to the context's code (append new definitions, lemmas, etc.).
        """
        if self.code:
            self.code += "\n" + new_code
        else:
            self.code = new_code

    def __repr__(self) -> str:
        context_str = "Context("
        context_str += f"is_valid={self.is_valid}, "
        context_str += f"id={self.id}, "
        context_str += f"code={[self.code]}, "
        context_str += f"states={[state.id for state in self.states.values()]}, "
        if not self.is_valid:
            context_str += f"messages={self.messages}, "
        context_str += ")"
        return context_str
