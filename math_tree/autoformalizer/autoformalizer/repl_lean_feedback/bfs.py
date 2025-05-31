import random
import time

from leanrepl import LeanREPL


def value_function(state):
    return 0.5


def policy_function(state):
    tactics = ["apply?", "rw?", "simp", "linarith", "norm_num", "ring", "aesop"]
    probs = [1 / len(tactics)] * len(tactics)
    return tactics, probs


class BestFirstSearch:
    def __init__(
        self,
        repl,
        context,
        initial_state,
        policy_function,
        value_function,
        k=3,
        time_limit=10,
        visits_limit=100,
        verbose=False,
    ):
        self.repl = repl
        self.context = context
        self.initial_state = initial_state
        self.policy_function = policy_function
        self.value_function = value_function
        self.k = k  # Number of suggested tactics to take into account when applying 'rw?' or 'apply?'
        self.time_limit = time_limit
        self.visits_limit = visits_limit
        self.verbose = verbose

        # Map from states to their values
        self.state_values = {initial_state: self.value_function(initial_state)}
        # Current state to expand
        self.current_state = initial_state

    def select_best(self):
        """
        Select the state with the highest heuristic value.
        """
        # 1. Find the maximum value
        max_value = max(self.state_values.values())

        # 2. Find the states with this maximum value
        best_states = [
            state for state, value in self.state_values.items() if value == max_value
        ]

        # 3. Select a random state among the best states
        selected_state = random.choice(best_states)

        return selected_state

    def expand(self, state):
        """
        Apply a tactic to the current state to generate a new state.
        """
        tactics, probs = self.policy_function(state)
        selected_tactic = random.choices(tactics, weights=probs, k=1)[0]

        if self.verbose:
            print("Selected tactic:", selected_tactic)

        if selected_tactic in ["rw?", "apply?"]:
            # Apply the tactic and parse the suggested tactics
            tmp_state, _ = self.repl.apply_tactic(self.context, state, selected_tactic)
            suggested_tactics = self._handle_suggested_tactics(tmp_state)

            if not suggested_tactics:
                return

            for suggested_tactic in suggested_tactics:
                if self.verbose:
                    print("Suggested tactic:", suggested_tactic)

                # Apply the tactic
                new_state, _ = self.repl.apply_tactic(
                    self.context, state, suggested_tactic, state
                )

                if new_state.is_valid:
                    # Count it as a visit
                    self.visits += 1
                    # Evaluate the new state
                    new_state_value = self.value_function(new_state)
                    self.state_values[new_state] = new_state_value

                    if self.verbose:
                        print("New goals:", new_state.goals)

                    if new_state.is_solved():
                        return new_state

                else:
                    if self.verbose:
                        print("Invalid state.")

                return

        else:
            # Apply the tactic
            new_state, _ = self.repl.apply_tactic(self.context, state, selected_tactic)

            if new_state.is_valid:
                # Count it as a visit
                self.visits += 1
                # Evaluate the new state
                new_state_value = self.value_function(new_state)
                self.state_values[new_state] = new_state_value

                if self.verbose:
                    print("New goals:", new_state.goals)

                if new_state.is_solved():
                    return new_state

                return new_state

            else:
                if self.verbose:
                    print("Invalid state.")

        return

    def run(self):
        """
        Execute the best-first search algorithm.
        """
        self.start_time = time.time()
        self.visits = 1

        if self.current_state.is_solved():
            print("Initial state is already solved.")
            return self.current_state

        while (
            time.time() - self.start_time < self.time_limit
            and self.visits < self.visits_limit
        ):
            # Select the best state
            self.current_state = self.select_best()
            if self.verbose:
                print("---\nCurrent goals: {}".format(self.current_state.goals))

            # Expand the current state
            new_state = self.expand(self.current_state)

            if new_state and new_state.is_solved():
                print(f"Solution found in {self.visits} visits!")
                return new_state

        print("Time limit or node limit reached. No solution found.")
        # Print the number of states in the state_values dictionary
        print(f"Number of unique states visited: {len(self.state_values)}")
        return None

    def _handle_suggested_tactics(self, state):
        """
        Extract the first k suggested tactics from the 'rw?' tactic.
        """
        suggested_tactics = []

        for msg in state.messages:
            try:
                message_text = msg.get("message", "")
                if message_text.startswith("Try this: "):
                    tactic = message_text[len("Try this: ") :].strip()
                    tactic = tactic.split("\n--")[0].strip()
                    if tactic not in suggested_tactics:
                        suggested_tactics.append(tactic)
                    if len(suggested_tactics) >= self.k:
                        break
            except Exception as e:
                print(f"Error handling suggested tactics : {e}")
                continue

        return suggested_tactics


if __name__ == "__main__":
    repl = LeanREPL()

    print("Importing libraries...")
    context = repl.create_context(
        """
                                import Mathlib.Data.Real.Basic
                                import Mathlib.Tactic"""
    )
    print("Libraries imported.")

    theorem_statement = """theorem algebra_9472 (x : ℝ) (hx : x^2 - 6 * x + 5 = 0) : x = 1 ∨ x = 5 := by sorry"""

    initial_state = repl.start_theorem(theorem_statement.strip(), context)

    print("---\nBest-first search:")
    bfs = BestFirstSearch(
        repl,
        context,
        initial_state,
        policy_function,
        value_function,
        time_limit=20,
        visits_limit=50,
        verbose=True,
    )
    state = bfs.run()
    if state:
        print("---\nProof sequence:")
        proof_sequence = repl.reconstruct_proof(context, state)
        print("---\nContext:")
        print(proof_sequence["context"])
        print("---\nStatement:")
        print(proof_sequence["statement"])
        print("---\nSteps:")
        for step in proof_sequence["steps"]:
            print(step)

    # Closing the REPL process
    repl.close()
