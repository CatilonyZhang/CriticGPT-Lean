import json
import os
import subprocess
import threading
import time
import traceback
from typing import Union

from autoformalizer.eval_utils.constants import path_to_mathlib, path_to_repl
from autoformalizer.repl_lean_feedback.context import Context
from autoformalizer.repl_lean_feedback.state import State


class LeanREPL:
    def __init__(self, path_to_repo=path_to_mathlib):
        # Start the REPL process using 'lake env {path_to_repl}'
        self.process = subprocess.Popen(
            ["lake", "env", path_to_repl],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
            cwd=path_to_repo,  # Set the working directory to given repository, default to `mathlib4`
            env=os.environ,  # Inherit environment variables
        )
        # Create a lock for thread safety
        self.lock = threading.Lock()
        # Dictionary to store Context objects indexed by context_id, for debugging
        self.contexts = {}

    def _send_command(self, command: dict) -> dict:
        """
        Send a JSON command to the REPL and return the JSON response.
        """
        with self.lock:
            # Convert the command to JSON and add two newlines
            json_command = json.dumps(command, ensure_ascii=False) + "\n\n"
            # Send the command to the REPL
            self.process.stdin.write(json_command)
            self.process.stdin.flush()

            # Read the response until a blank line is encountered
            response_lines = []
            try:
                while True:
                    line = self.process.stdout.readline()
                    if line.strip() == "":
                        # Blank line indicates the end of the response
                        break
                    response_lines.append(line)
            except Exception as e:
                print("Error reading response:", e)
                print("traceback:", traceback.format_exc())
                return {}

            # Combine the response lines and parse the JSON
            response_str = "".join(response_lines)
            try:
                response_json = json.loads(response_str)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                print("Response received:", response_str)
                response_json = {}
            return response_json

    def create_context(self, code: str) -> Union[Context, None]:
        """
        Send code to create a new context.
        """
        command = {"cmd": code}
        response = self._send_command(command)
        context_id = response.get("env")
        if context_id is not None:
            context = Context(id=context_id, code=code)
            self.contexts[context_id] = context
            return context
        else:
            print("Failed to create context")
            return None

    def extend_context(self, context: Context, code: str) -> Context:
        """
        Add code (definitions, lemmas, etc.) to an existing context.
        """
        command = {"cmd": code, "env": context.id}
        response = self._send_command(command)
        new_context_id = response.get("env")
        if new_context_id is not None:
            new_code = context.code + "\n" + code
            new_context = Context(id=new_context_id, code=new_code)
            return new_context
        else:
            print("Failed to create context")
            return None

    def start_theorem(
        self, statement: str, context: Context = None
    ) -> Union[State, None]:
        """
        Initialize a new theorem to prove.
        """
        if context is None:
            print("Please specify a context in which to start the theorem.")
            return None
        else:
            command = {"cmd": statement, "env": context.id}
        response = self._send_command(command)
        state = State.from_response(response, statement=statement)
        if state is not None and state.id is not None:
            # Store the new state in the context
            context.states[state.id] = state
        return state

    def apply_tactic(
        self, context: Context, state: State, tactic: str, parent_state: State = None
    ) -> Union[State, None]:
        """
        Apply a tactic to a specific proof state identifier.
        """
        # Before applying the tactic, check if the state is valid
        if not state.is_valid:
            print("Cannot apply tactic to an invalid state.")
            return None

        # Define the parent state
        if not parent_state:
            parent_state = state

        # Send the tactic command to the REPL
        command = {"tactic": tactic, "proofState": state.id}
        start_time = time.time()
        response = self._send_command(command)
        end_time = time.time()
        time_taken = end_time - start_time

        # Create a new state from the response
        new_state = State.from_response(
            response,
            parent_state=parent_state,
            last_tactic=tactic,
            statement=state.statement,
        )

        # Store the new state in the context if it is valid
        if new_state.is_valid and tactic not in ["apply?", "rw?"]:
            context.states[new_state.id] = new_state
            state.children.append(new_state)

        return new_state, time_taken

    def reconstruct_proof(self, context: Context, state: State) -> dict:
        """
        Reconstruct the proof sequence from the initial state to the given state.
        """
        proof_steps = []
        current_state = state

        step = {"goals": current_state.goals}
        proof_steps.insert(0, step)
        # Traverse back to the initial state
        while current_state:
            if current_state.parent_state is not None:
                step = {
                    "goals": current_state.parent_state.goals,
                    "tactic": current_state.tactics[-1],
                }
                proof_steps.insert(0, step)
                current_state = current_state.parent_state
            else:
                if current_state.statement:
                    statement = current_state.statement
                break

        return {"context": context.code, "statement": statement, "steps": proof_steps}

    def extract_infotree_from_file(self, file_path: str) -> Union[dict, None]:
        """
        Extract the infotree from a file

        param path: the path to the lean file
        return: repl response
        """
        command = {"path": file_path, "infotree": "original"}
        response = self._send_command(command)
        context_id = response.get("env")
        if context_id is not None:
            context = Context(id=context_id, code="")
            self.contexts[context_id] = context
            return response
        else:
            print(f"Failed to create context for file {file_path}")
            return None

    def pickle_env(self, context_id: int, output_path: str) -> Union[str, None]:
        """
        Pickles the context into an .olean file according to the path for quick access later on

        :param context_id: the context id to pickle
        :param path: the path to pickle the context to
        :return: The full path of the saved .olean file if successful, or None if the operation fails.
        """
        if not output_path.endswith(".olean"):
            olean_path = f"{output_path}.olean"
        command = {"pickleTo": olean_path, "env": context_id}
        response = self._send_command(command)
        context_id = response.get("env")
        if context_id is not None:
            return olean_path
        else:
            print("Failed to pickle: context does not exist")
            return None

    def pickle_state(self, state_id: int, output_path: str) -> Union[str, None]:
        """
        Pickles the proof state into an .olean file according to the path for quick access later on

        :param state_id: the proof state id to pickle
        :param path: the path to pickle the proof state to
        :return: The full path of the saved .olean file if successful, or None if the operation fails.
        """
        if not output_path.endswith(".olean"):
            olean_path = f"{output_path}.olean"
        command = {"pickleTo": olean_path, "proofState": state_id}
        response = self._send_command(command)
        new_state_id = response.get("proofState")
        if new_state_id is not None:
            return olean_path
        else:
            print("Failed to pickle: context does not exist")
            return None

    def pickle_code(self, code: str, output_path: str):
        """
        Pickles the input lean4 code into an .olean file according to the path for quick access later on
        The actual olean file is saved at the {path_to_mathlib4}/output_path.olean

        :param lean_code: The Lean 4 code to be pickled.
        :param output_path: The file path (excluding extension) where the .olean file will be saved.
        :return: The full path of the saved .olean file if successful, or None if the operation fails.
        """
        context = self.create_context(code)
        if context is not None:
            context_id = context.id
            olean_path = self.pickle_env(context_id, output_path)
            if olean_path is not None:
                return olean_path
        print("Failed to create context")
        return None

    def unpickle_env(self, olean_path: str) -> Union[int, None]:
        """
        Unpickles the context from an .olean file according to the path for quick access later on

        :param path: the path to the .olean file to unpickle
        :return: The context id if successful, or None if the operation fails.
        """
        command = {"unpickleEnvFrom": olean_path}
        response = self._send_command(command)
        context_id = response.get("env")
        if context_id is not None:
            return context_id
        else:
            print("Failed to unpickle the environment")
            return None

    def unpickle_state(self, olean_path: str) -> Union[int, None]:
        """
        Unpickles the proof state from an .olean file according to the path for quick access later on

        :param path: the path to the .olean file to unpickle
        :return: The proof state id if successful, or None if the operation fails
        """
        command = {"unpickleProofStateFrom": olean_path}
        response = self._send_command(command)
        state_id = response.get("proofState")
        if state_id is not None:
            return state_id
        else:
            print("Failed to unpickle the environment")
            return None

    def close(self):
        """
        Terminate the REPL process gracefully.
        """
        self.process.terminate()
        self.process.wait()
