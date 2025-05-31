from autoformalizer.repl_lean_feedback.leanrepl import LeanREPL

# Example usage
if __name__ == "__main__":
    # Create a Lean REPL process
    repl = LeanREPL()
    print("New LeanREPL created.")
    try:
        # Create a context: import libraries, write definitions, write useful lemmas, etc.
        context = repl.create_context("import Mathlib")
        print("---\nNew Context created with:\nimport Mathlib")

        # Start a theorem
        theorem_statement = """theorem algebra_9472 (x : ℝ) (hx : x^2 - 6 * x + 5 = 0) : x = 1 ∨ x = 5 := by sorry"""
        state = repl.start_theorem(theorem_statement.strip(), context)
        print("---\nNew theorem initialized with:\ntheorem algebra_9472 (x : ℝ) (hx : x^2 - 6 * x + 5 = 0) : x = 1 ∨ x = 5")
        if not state.is_valid:
            print('Initial state is not valid.')

        # List of tactics that are going to be applied
        # Note that we don't need to make a list of tactics, we can just apply them one by one
        tactics = [
            "have: x ^ 2 - 6 * x + 5 = (x-5)*(x-1) := by ring",
            "rw [this] at hx",
            "have eq': x-5 = 0 ∨ x - 1 = 0 := mul_eq_zero.mp hx",
            "rcases eq' with e5 | e1",
            "right",
            "rw [← add_zero 5, ← e5]",
            "simp",
            "left",
            "rw [← add_zero 1, ← e1]",
            "simp",
        ]

        # Initialising
        times = []
        total_time = 0

        for tactic in tactics:
            print(
                f"---\nCurrent goals: {state.goals}\nTactic: {tactic}"
            )  # Printing the current goal states and the tactic that we are going to apply
            new_state, time_taken = repl.apply_tactic(
                context, state, tactic
            )  # Applying the tactic and getting Lean4 feedback
            print(f"New goals: {new_state.goals}")  # Printing the new goal states
            state = new_state
            # Checking if the state is valid (ie. not an error state)
            if not state.is_valid:
                print("---\nInvalid state. Stopping proof.")
                break
            times.append(time_taken)
            total_time += time_taken

        # After applying the tactics, we check if the state is solved (ie. if the proof is complete)
        if state.is_solved():
            print("---\nProof is complete!")
        else:
            print("---\nProof is not yet complete.")

        # Printing the average time taken per tactic to send to and get feedback from Lean4
        average_time = total_time / len(tactics)
        print(f"---\nAverage time per tactic: {average_time:.3f} seconds")
        print(f"Total time for the proof: {total_time:.3f} seconds")

        # We can reconstruct the whole proof sequence
        # with the context, statement, intermediate goal states and tactics applied
        print("---\nProof sequence:")
        proof_sequence = repl.reconstruct_proof(context, state)

        print("---\nContext:")
        print(proof_sequence["context"])

        print("---\nStatement:")
        print(proof_sequence["statement"])

        print("---\nSteps:")
        for step in proof_sequence["steps"]:
            print(step)

    finally:
        # Closing the REPL process
        repl.close()
