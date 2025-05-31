import json
import pathlib
import uuid
from typing import List

from autoformalizer.repl_lean_feedback.repl_utils import (
    parse_error_message,
    parse_messages,
)


class State:
    def __init__(
        self,
        id: str = None,
        goals: List[str] = None,
        messages: dict = None,
        is_valid: bool = None,
        parent_state: "State" = None,
        tactics: List[str] = None,
        statement: str = None,
        full_code: str = None,
        full_code_lean_response: dict = None,
    ):
        self.id = id  # Identifier (int)
        self.goals = goals or []  # List of goals (strings)
        self.messages = messages or []  # List of messages (dicts)
        self.parent_state = parent_state  # Parent state (State)
        self.tactics = tactics  # Previous tactic applied to reach this state (string)
        self.statement = statement  # Theorem statement (str)
        self.is_valid = is_valid  # Whether the state is valid (bool)
        self.full_code = full_code
        self.full_code_lean_response = full_code_lean_response

    def is_solved(self):
        """
        Check if the state has no remaining goals.
        """
        return self.is_valid and not self.goals

    def to_dict(state):
        """Convert State object to dictionary."""
        return {
            "id": state.id,
            "goals": state.goals,
            "messages": state.messages,
            "is_valid": state.is_valid,
            "parent_state_id": state.parent_state.id if state.parent_state else None,
            "tactics": state.tactics,
            "statement": state.statement,
            "full_code": state.full_code,
            "full_code_lean_response": state.full_code_lean_response,
        }

    def from_dict(state_dict):
        """Convert dictionary to State object."""
        return State(
            id=state_dict["id"],
            goals=state_dict["goals"],
            messages=state_dict["messages"],
            is_valid=state_dict["is_valid"],
            parent_state=state_dict["parent_state_id"],
            tactics=state_dict["tactics"],
            statement=state_dict["statement"],
            full_code=state_dict["full_code"],
            full_code_lean_response=state_dict["full_code_lean_response"],
        )

    @classmethod
    def from_seq_apply(
        cls, messages, goals, parent_state=None, last_tactic=None, statement=None
    ):
        messages = messages or []
        if all(msg["severity"] != "error" for msg in messages):
            is_valid = True
        else:
            is_valid = False

        if parent_state and parent_state.tactics:
            tactics = parent_state.tactics.copy()
        else:
            tactics = []

        if last_tactic:
            tactics.append(last_tactic)

        return cls(
            id=str(uuid.uuid4()),
            goals=goals,
            messages=messages,
            parent_state=parent_state,
            tactics=tactics,
            statement=statement,
            is_valid=is_valid,
        )

    @classmethod
    def from_response(
        cls, response, parent_state=None, last_tactic=None, statement=None
    ):
        # 1) Get the messages from the response
        if "messages" in response:
            messages = parse_messages(response.get("messages", []))
        elif "message" in response:
            messages = parse_error_message(response.get("message", ""))
        else:
            messages = []

        # 2) Check if the state is valid
        if ("sorries" in response or "proofState" in response) and all(
            msg["severity"] != "error" for msg in messages
        ):
            is_valid = True
        else:
            is_valid = False

        # 3) Build tactics list by appending last_tactic to parent's tactics
        if parent_state and parent_state.tactics:
            tactics = parent_state.tactics.copy()
        else:
            tactics = []

        if last_tactic:
            tactics.append(last_tactic)

        # 4) Get the id and goals from the response
        if "sorries" in response:
            proof_state_info = response["sorries"][0]
            id = proof_state_info.get("proofState", "")
            goal = proof_state_info.get("goal", "")
            goals = [goal]
        elif "proofState" in response:
            id = response.get("proofState", "")
            goals = response.get("goals", [])
        else:
            id = None
            goals = None

        # 5) Return the state
        return cls(
            id=id,
            goals=goals,
            messages=messages,
            parent_state=parent_state,
            tactics=tactics,
            statement=statement,
            is_valid=is_valid,
        )

    def __repr__(self) -> str:
        state_str = "State("
        state_str += f"id={self.id}, "
        state_str += f"is_valid={self.is_valid}, "
        if self.is_valid:
            state_str += f"goals={self.goals}, "
        else:
            state_str += f"messages={self.messages}, "
        if self.parent_state:
            state_str += f"parent_state_id={self.parent_state.id}, "
        else:
            state_str += f"parent_state={None}, "
        state_str += f"tactics={self.tactics}, "
        state_str += ")"
        return state_str

    def __eq__(self, other):
        if str(self.goals) + str(self.is_valid) != str(other.goals) + str(
            other.is_valid
        ):
            return False
        return True

    def __hash__(self):
        return hash(str(self.goals) + str(self.is_valid))


def serialize_states(all_states, file_path: pathlib.Path, serialized=None):
    lines = []
    for state in all_states:
        if state.id not in serialized:
            state_dict = state.to_dict()
            serialized.add(state.id)
            lines.append(json.dumps(state_dict))

    with open(file_path, "a") as f:
        for line in lines:
            f.write(line + "\n")
    return serialized


def deserialize_states(file_path: pathlib.Path):
    """
    Deserialize states from a JSON file and rebuild the state hierarchy.

    :param filename: The filename to load the JSON data from (default is 'states.json')
    :return: A tuple containing the root_state (State) and a list of all states (List[State])
    """
    if not file_path.exists():
        return None

    all_states = {}
    with open(file_path, "r") as f:
        for line in f:
            state_dict = json.loads(line.strip())
            state = State.from_dict(state_dict)
            all_states[state.id] = state

    # Now, assign parent_state to each state (after all states are loaded)
    for state in all_states.values():
        if isinstance(state.parent_state, str):
            state.parent_state = all_states.get(state.parent_state)

    return all_states


if __name__ == "__main__":
    response_1 = {
        "proofState": 22,
        "messages": [
            {
                "severity": "error",
                "pos": {"line": 0, "column": 0},
                "endPos": {"line": 0, "column": 0},
                "data": "unsolved goals\nx y : ℤ\nh : x ^ 5 =",
            }
        ],
        "goals": ["x y : ℤ\nh : x ^ 5 = y ^ 2 + 4\n"],
    }

    response_2 = {
        "message": "Lean error:\n<input>:1:26: unexpected end of input; expected '{'"
    }

    state = State.from_response(response_2)
