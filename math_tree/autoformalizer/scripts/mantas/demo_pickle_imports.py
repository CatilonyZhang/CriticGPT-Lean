import time

from autoformalizer.repl_lean_feedback.leanrepl import LeanREPL

def pickle_imports(imports, path):
    repl = LeanREPL()
    olean_path = repl.pickle_code(imports, path)
    return olean_path

def unpickle_path(olean_path):
    repl = LeanREPL()
    env = repl.unpickle_env(olean_path)
    return env
    

if __name__ == '__main__':
    # imports = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
    imports = "import Mathlib"
    path = "imports"
    reduction = []
    # Test the pickle/unpickle
    for i in range(10):
        print("Iteration: ", i)
        time_0 = time.time()
        olean_path = pickle_imports(imports, path)
        time_1 = time.time()
        env = unpickle_path(olean_path)
        time_2 = time.time()
        time_taken_pickle = time_1 - time_0
        time_taken_unpickle = time_2 - time_1
        reduction.append((time_taken_pickle - time_taken_unpickle) * 100 / time_taken_pickle)
    print(f"Import statement : {{imports}} \n Percentage speedup: ", sum(reduction) / len(reduction))