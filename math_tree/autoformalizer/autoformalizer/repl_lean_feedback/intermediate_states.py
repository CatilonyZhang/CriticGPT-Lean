import os
import re
import tempfile

from autoformalizer.eval_utils.constants import path_to_mathlib
from autoformalizer.repl_lean_feedback.leanrepl import LeanREPL


class TacticSnapshot:
    def __init__(self, tactic, goals_before, goals_after, start, finish, comment=""):
        self.tactic = tactic  # tactic applied (string)
        self.goals_before = (
            goals_before or []
        )  # List of goals before tactic call (strings)
        self.goals_after = goals_after or []  # List goals after tactic call (strings)
        self.start = start  # Start position of given tactic (string)
        self.finish = finish  # Finish position of given tactic (string)
        self.comment = comment or ""  # End position of given tactic (string)

    def __str__(self):
        """
        Provides a string representation of the TacticSnapshot instance.
        """
        return (
            f"Tactic: [{self.tactic}] \n"
            f"Goals Before: [{self.goals_before}] \n"
            f"Goals After: [{self.goals_after}] \n"
            f"Start: [{self.start}] \n"
            f"Finish: [{self.finish}] \n"
            f"Comment: [{self.comment}] \n"
        )

    def to_dict(self):
        """
        Provides a dictionary representation of the TacticSnapshot instance.
        This is used when creating a dataset, and does not include the start and finish positions.
        """
        return {
            "tactic": self.tactic,
            "position": {"start": self.start, "finish": self.finish},
            "goals_before": self.goals_before,
            "goals_after": self.goals_after,
            "comment": self.comment,
        }


class TheoremSnapshot:
    def __init__(
        self,
        header,
        context,
        statement,
        steps,
        comment="",
        folder=None,
        file=None,
        has_sorry=None,
    ):
        self.folder = folder  # folder
        self.file = file  # file
        self.header = header  # header of the file (string)
        self.context = context  # file context of the theorem (string)
        self.statement = statement  # theorem statement (string)
        self.steps = steps  # proof steps coming from the file, proving the theorem (list of TacticSnapshot)
        self.comment = comment  # theorem comment if there exists one (string)
        self.has_sorry = (
            has_sorry  # indicating whether the proof contains a "sorry" (bool)
        )

    def __str__(self):
        """
        Provides a string representation of the TheoremSnapshot instance.
        """
        return (
            f"Folder: {self.folder}\n"
            f"File: {self.file}\n"
            f"Header: {self.header}\n"
            f"Context: {self.context}\n"
            f"Statement: {self.statement}\n"
            f"Steps: [{self.steps}]\n"
            f"Comment: {self.comment}\n"
        )

    def to_dict(self):
        """
        Provides a dictionary representation of the TheoremSnapshot instance.
        """
        return {
            "folder": self.folder,
            "file": self.file,
            "header": self.header,
            "context": self.context,
            "statement": self.statement,
            "steps": self.steps,
            "comment": self.comment,
            "has_sorry": self.has_sorry,
        }


def extract_all_nodes(data):
    """
    Recursively extracts all 'node' information from a list of dictionaries or nested dictionaries.

    :param data: List of dictionaries or a single dictionary with keys 'node', 'kind', 'children'.
    :return: A list of all 'node' values.
    """
    nodes = []

    # If the input is a dictionary, convert it to a list for uniform handling
    if isinstance(data, dict):
        data = [data]

    # Iterate over each dictionary in the list
    for item in data:
        if "node" in item:
            nodes.append(item["node"])  # Extract the 'node' value

        # If the item contains 'children', recursively process them
        if "children" in item and isinstance(item["children"], list):
            nodes.extend(extract_all_nodes(item["children"]))

    return nodes


def extract_leaf_nodes(data):
    """
    Recursively extracts all **leaf** 'node' information from a list of dictionaries or nested dictionaries.
    This is useful when only single tactic steps are extracted.

    :param data: List of dictionaries or a single dictionary with keys 'node', 'kind', 'children'.
    :return: A list of all **leaf** 'node' values.
    """
    nodes = []

    # If the input is a dictionary, convert it to a list for uniform handling
    if isinstance(data, dict):
        data = [data]

    # Iterate over each dictionary in the list
    for item in data:

        # If the item contains non-synthetic 'children', recursively process them
        if (
            "children" in item
            and isinstance(item["children"], list)
            and item["children"] != []
        ):
            has_child = False  # boolean to keep track of if the current node has a non-synthetic child
            for child in item["children"]:
                if child["node"]["stx"]["range"]["synthetic"] is False:
                    has_child = True
                    nodes.extend(extract_leaf_nodes(child))
            # If current node has no non-synthetic children, then it is a leaf node, so add it to the array
            if has_child is False:
                nodes.append(item["node"])  # Extract the leaf 'node' value
        # If current node is a leaf node, add it to the array
        else:
            nodes.append(item["node"])

    return nodes


def extract_parent_nodes(data):
    """
    Recursively extract the parent 'node' information from a list of dictionaries or nested dictionaries.
    This is useful when only single tactic steps are extracted.

    :param data: List of dictionaries or a single dictionary with keys 'node', 'kind', 'children'.
    :return: A list of all **leaf** 'node' values.
    """
    nodes = []

    # Iterate over each dictionary in the list
    for item in data:
        nodes.append(item["node"])  # Extract the parent 'node' value

    return nodes


def extract_tactic_snapshot(node):
    """
    Process 'node' information and output a TacticSnapshot object

    :param data: 'node' information coming from Lean's InfoTree structure
        including fields 'stx', 'pp', 'goals_before', 'goals_after' and etc.
    :return: a TacticSnapshot object
    """
    stx = node["stx"]
    stx_range = stx["range"]
    goals_before = node["goalsBefore"]
    goals_after = node["goalsAfter"]
    if "synthetic" in stx_range:
        if stx_range["synthetic"] is False:
            start = stx_range["start"]
            finish = stx_range["finish"]
            tactic = stx["pp"]
            if "<failed to pretty print>" not in tactic:
                tactic_snapshot = TacticSnapshot(
                    tactic, goals_before, goals_after, start, finish
                )
                return tactic_snapshot
    
    # Return None if the node is synthetic or the tactic could not be pretty printed
    return None


def get_header(file_path):
    """
    Function to get header of the file from the input file path
    """
    import_lines = []
    open_lines = []
    toggle = False
    with open(file_path, "r") as file:

        for line in file:
            if line.startswith("import"):
                if toggle is False:
                    toggle = True
                import_lines.append(
                    line
                )  # Add the line to the list, stripping whitespace
            else:
                if toggle is True:
                    toggle = False
                    break  # Stop reading when a line doesn't start with 'import'
        for line in file:
            if line.startswith("open"):
                if toggle is False:
                    toggle = True
                open_lines.append(
                    line
                )  # Add the line to the list, stripping whitespace
            else:
                if toggle is True:
                    break  # Stop reading when a line doesn't start with 'open'
    # check that we are not at the end of the file
    imports = "".join(import_lines)
    opens = "".join(open_lines)
    header = imports + "\n" + opens
    return header


def get_context(file_path, header, tactic_snapshot: TacticSnapshot):
    """
    Function to get context of a TacticSnapshot of the file from the input file path
    beginning after the header and ending with the current tactic call
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Step 1: Strip empty lines and match header lines
    header_lines = header.splitlines()
    header_lines = [line.strip() for line in header_lines]
    i = 0  # Line index
    # Find the line in the file matching the last non-white-space line in the header
    while i < len(lines) and lines[i].strip() != header_lines[-1]:
        i += 1

    start = tactic_snapshot.start
    target_line = start["line"]
    target_column = start["column"]

    context = ""
    i += 1
    while i < len(lines) and i + 1 <= target_line:
        line_content = lines[i]
        if i + 1 == target_line:  # Stop at the target line
            context += line_content[:target_column]  # Include up to the target column
            break
        context += line_content
        i += 1

    return context


def extract_comments(context):
    """
    Extract comments from the context string
    """
    # Split the text block into lines
    lines = context.splitlines()

    # Initialize an empty list to store the comment lines
    comment_lines = []

    # Iterate backwards through the lines
    for i in range(len(lines) - 1, -1, -1):  #
        line = lines[i]

        if line.strip() == "":
            continue
        # Check for single-line comment
        if line.lstrip().startswith("--"):
            comment_lines.insert(0, line)

        # Check for multi-line comment end
        else:
            if line.rstrip().endswith("-/") or line.rstrip().endswith("--/"):
                # Iterate backwards to find the start of the multi-line comment
                for j in range(i, -1, -1):
                    prev_line = lines[j]
                    if prev_line.lstrip().startswith(
                        "/-"
                    ) or prev_line.lstrip().startswith("/--"):
                        comment_lines.insert(0, prev_line)
                        break
                    comment_lines.insert(0, prev_line)
            else:
                break

    if comment_lines != []:
        # Join the comment lines into a single string
        comment = "\n".join(comment_lines)
        return comment
    else:
        return None


def extract_theorem_context(context, theorem_statement):
    trimmed_statement = theorem_statement[:-11]
    # Find the last occurrence of the substring
    last_occurrence_index = context.rfind(trimmed_statement)

    # Check if the substring is found
    if last_occurrence_index != -1:
        # Trim the text until the last occurrence of the substring
        return context[:last_occurrence_index]
    else:
        # If the substring is not found, return the original text
        return context


def remove_comments(tactic_snap: TacticSnapshot):
    """
    Removes comments from the tacticSnapshots tactic string
    """
    tactic = tactic_snap.tactic
    single_line_comment_pattern = r"--.*"
    multi_line_comment_pattern = r"/-.*?-/"
    multi_line_comment_pattern_alt = r"/--.*?--/"

    # Combine patterns into a single regex with named groups
    combined_pattern = re.compile(
        r"(?P<single_line>"
        + single_line_comment_pattern
        + r")|"
        + r"(?P<multi_line>"
        + multi_line_comment_pattern
        + r")|"
        + r"(?P<multi_line_alt>"
        + multi_line_comment_pattern_alt
        + r")",
        re.DOTALL,
    )

    # Remove all comments
    text_without_comments = combined_pattern.sub("", tactic)

    # Split the text into lines and remove empty lines
    lines = text_without_comments.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]

    # Join the non-empty lines back into a single string
    cleaned_tactic = "\n".join(non_empty_lines)
    tactic_snap.tactic = cleaned_tactic
    return tactic_snap

def extract_statement(context):
    """
    Extract theorem statement from context string
    """
    # regular expression pattern to extract theorem statements
    matches = list(
        re.finditer(
            r"^(?:(private|nonrec|private nonrec)\s+)?(theorem|lemma|example)\b",
            context,
            re.MULTILINE,
        )
    )

    matches_def = list(
        re.finditer(r"^\s*(?:private\s+|noncomputable\s+)*def", context, re.MULTILINE)
    )

    if matches:
        last_match = matches[-1]
        # check if the last declaration is a def
        if matches_def:
            last_def = matches_def[-1]
            if last_match.start() < last_def.start():
                return None
        start_position = last_match.start()
        stop_pattern = r":=\s*(?:by\s*)?\n"
        stop_match = re.search(stop_pattern, context[start_position:])
        if stop_match:
            return (
                context[start_position : start_position + stop_match.start()]
                + ":= by sorry"
            )
        else:
            return None
    else:
        return None  # Return None if no "theorem/lemma/example" was found
    
def extract_compressed_lemma_and_theorems(trees):
    """
    Extracts theorems and lemmas from the infotree

    TODO: IMPORTANT: This function is not working as expected and needs to be fixed
    """
    statements = []
    for tree in trees:
        try:
            tree_pp = tree["node"]["stx"]["pp"]
            if tree_pp is None:
                continue
            if tree_pp.startswith("theorem") or tree_pp.startswith("lemma") or tree_pp.startswith("example"):
                theorem_text = "".join(tree_pp.split())

                # the proof content will always be the second last child
                
                child = tree["children"][-2]
                proof_pp = "".join(child["node"]["stx"]["pp"].split())
                assert proof_pp in theorem_text
                theorem_statment = theorem_text[:theorem_text.index(proof_pp)]
                statements.append((theorem_statment, proof_pp, theorem_text))
        except KeyError:
            continue
        except IndexError:
            print(trees)
    return statements

def extract_proof_input_output(lean_feedback, lean_code):
    """
    Extracts the proof input and output from the proof code

    TODO: IMPORTANT: This function is not working as expected and needs to be fixed
    """
    def _extract_proof_input_output(statements, code):
        """
        Extracts the proof input and output from the proof code
        """
        proof_input_and_outputs = []
        for statement in statements:
            packed_statement, packed_proof, _ = statement
            if packed_proof.startswith("by"):
                packed_statement += "by"

            # Compress code by removing all whitespace characters
            compressed_code = ''.join(code.split())

            # Find the starting index of B in the compressed A
            index = compressed_code.find(packed_statement)
            if index == -1:
                print(f"Statement: {packed_statement}")
                print("Code:", code)
                raise ValueError("Compressed string B not found in A")

            # Determine the end position of the matched B in compressed A
            end_compressed = index + len(packed_statement)

            # Map the end_compressed position back to the original A's index
            compressed_count = 0
            split_point = None
            for i, c in enumerate(code):
                if not c.isspace():
                    compressed_count += 1
                    if compressed_count == end_compressed:
                        split_point = i + 1  # Include the current character
                        break

            # If the split point wasn't found within A, set it to the end of A
            if split_point is None:
                split_point = len(code)

            # Split A into before and after
            proof_input = code[:split_point]
            proof_output = code[split_point:]

            assert proof_input[-1] == packed_statement[-1]
            proof_input_and_outputs.append((proof_input, proof_output))

        return proof_input_and_outputs
    
    if "response" not in lean_feedback or lean_feedback["response"] is None:
        return None

    trees = lean_feedback["response"].get("infotree", None)
    if trees is None:
        return None

    statements = extract_compressed_lemma_and_theorems(trees)
    return _extract_proof_input_output(statements, lean_code)


def process_file(path, path_to_repo=path_to_mathlib, mode="leaf"):
    """
    Given a path of a file, process it returning a List of 'TheoremSnapshot' objects detailing the intermediate
    tactic states seens in the file

    :param data:
    'path' to a file.
        NOTE: this is currently taken from the `mathlib4` path where the REPL is initialized
    'mode' to specify if we want to extract all nodes, just the leaf nodes or also parent nodes,
        set to "leaf" by default
    :return:
    a List TacticSnapshot with intermediate tactic state information
    """
    # 1) Get the infotree
    repl: LeanREPL = LeanREPL(path_to_repo=path_to_repo)
    output = repl.extract_infotree_from_file(path)
    repl.close()
    # Check if the processed file has any infotrees
    if output is not None and "infotree" in output:
        trees = output["infotree"]
    else:
        # return empty list in case no infotrees are found
        return []
    if mode == "leaf":
        nodes = extract_leaf_nodes(trees)
    elif mode == "parent":
        nodes = extract_parent_nodes(trees)
    else:
        nodes = extract_all_nodes(trees)

    # 2) Create the tactic snaps from the nodes
    tactic_snaps = []
    for node in nodes:
        tactic_snapshot = extract_tactic_snapshot(node)
        if tactic_snapshot is not None:
            tactic_snaps.append(tactic_snapshot)
    # Get the header of a file
    header = get_header(file_path=path)
    # Traverse tactic snapshots to format then as we want them

    # 3) Group the tactic snaps in TheoremSnapshot objects
    theorem_snaps = {}
    for tactic_snap in tactic_snaps:
        tactic_snap = remove_comments(tactic_snap)
        context = get_context(
            file_path=path, header=header, tactic_snapshot=tactic_snap
        )  # To extract comments
        # slightly pre-processes if the parents were extracted
        if mode == "parent" and tactic_snap.tactic.lstrip().startswith("by\n"):
            context += " by\n"
            # remove "by \n" from tactic_snap.tactic from the beginning
            tactic_snap.tactic = tactic_snap.tactic.lstrip()[5:]
        tactic_snap.comment = extract_comments(context)
        statement = extract_statement(context)
        if statement is not None:
            if statement not in theorem_snaps:
                theorem_context = extract_theorem_context(context, statement)
                theorem_comment = extract_comments(theorem_context)
                theorem_snap = TheoremSnapshot(
                    header,
                    theorem_context,
                    statement,
                    [tactic_snap.to_dict()],
                    theorem_comment,
                )
                theorem_snaps[statement] = theorem_snap
            else:
                theorem_snaps[statement].steps.append(tactic_snap.to_dict())
    return list(theorem_snaps.values())


def process_code(code, mode):
    """
    Given a code snippet, process it returning a List of 'TacticSnapshot' objects detailing the intermediate
    tactic states seens in the code

    :param data: 'code' snippet.
    :return: a List TacticSnapshot with intermediate tactic state information
    """
    # create a TemporaryFile in mathlib4/tmp/
    directory = f"{path_to_mathlib}/tmp/"
    os.makedirs(directory, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, delete=False
    ) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        theorem_snaps = process_file(temp_file.name, mode=mode)

    os.remove(temp_file.name)

    return theorem_snaps
