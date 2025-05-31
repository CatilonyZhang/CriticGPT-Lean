def extract_nodes_and_edges(
    infotree, parent_id=None, start_id=0, include_failed_pp=True, deduplicate=False
):
    """
    Recursively extract nodes and edges from an infotree.
    This function is also useful for visualizing the infotree with `print_tree_interactive`.

    Parameters
    ----------
    infotree : list of dict
        A list of dictionaries each containing 'node' and optionally 'children'.
    parent_id : int or None, optional
        The ID of the parent node, or None if this is the root level (default is None).
    start_id : int, optional
        The next available integer ID to assign to a new node (default is 0).
    include_failed_pp : bool, optional
        If False, nodes whose pretty-print text is "<failed to pretty print>" are removed
        (default is True).
    deduplicate : bool, optional
        If True, deduplicate chains of identical nodes based on goalsBefore,
        goalsAfter, and pp (default is False).

    Returns
    -------
    nodes : dict of {int : dict}
        A dictionary of node_id -> node_content.
    edges : list of tuple
        A list of tuples (parent_id, child_id, {}) representing edges in the infotree.
    next_id : int
        The next available integer ID after processing all children.
    """
    nodes = {}
    edges = []

    current_id = start_id

    for item in infotree:
        if "node" in item:
            node_data = item["node"]

            # Add this node to the nodes dictionary
            node_id = current_id
            current_id += 1
            nodes[node_id] = node_data

            # Recursively handle the children of the node, depth-first approach
            if (
                "children" in item
                and isinstance(item["children"], list)
                and item["children"]
            ):
                child_nodes, child_edges, current_id = extract_nodes_and_edges(
                    item["children"],
                    parent_id=node_id,
                    start_id=current_id,
                    include_failed_pp=include_failed_pp,
                    deduplicate=deduplicate,
                )
                nodes.update(child_nodes)
                edges.extend(child_edges)

            # Add an edge from the parent to this node
            if parent_id is not None:
                edges.append((parent_id, node_id, {}))

            # Now handle possible flattening in a loop until there are no more changes
            transformed = True
            while transformed:
                transformed = False

                # 1) Remove/flatten all children that have failed PP (if include_failed_pp=False)
                #    We handle them individually, even if there's more than one child.
                #    Then we break to re-check children from scratch, as new failed PP children might appear.
                child_list = [
                    e for e in edges if e[0] == node_id
                ]  # edges from node_id to child
                for edge_obj in child_list:
                    child_id = edge_obj[1]
                    child_content = nodes.get(child_id, {})
                    child_pp = child_content.get("stx", {}).get("pp", "")

                    if not include_failed_pp and child_pp == "<failed to pretty print>":
                        # Flatten this child: remove it and connect parent directly to its children
                        # Note that the new children might also have failed PP, so we need to re-check them
                        _flatten_chain(nodes, edges, node_id, child_id)
                        transformed = True
                        break

                if transformed:
                    # We need to restart the while loop to re-check children from scratch, as there might be
                    # more failed PP children
                    continue

                # 2) Deduplicate if there's exactly one child left that has same (goalsBefore, goalsAfter, pp)
                # This happens quite often in infotrees, where a tactic is repeated multiple times, extracted
                # with different parsers
                child_list = [e for e in edges if e[0] == node_id]
                if deduplicate and len(child_list) == 1:
                    child_id = child_list[0][1]
                    if child_id in nodes:
                        child_content = nodes[child_id]
                        # Compare parent's vs child's (goalsBefore, goalsAfter, pp)
                        parent_goalsBefore = node_data.get("goalsBefore", [])
                        parent_goalsAfter = node_data.get("goalsAfter", [])
                        parent_pp = node_data.get("stx", {}).get("pp", "")

                        child_goalsBefore = child_content.get("goalsBefore", [])
                        child_goalsAfter = child_content.get("goalsAfter", [])
                        child_pp = child_content.get("stx", {}).get("pp", "")

                        if (
                            child_goalsBefore == parent_goalsBefore
                            and child_goalsAfter == parent_goalsAfter
                            and child_pp == parent_pp
                        ):
                            # Flatten this child: remove it and connect parent directly to its children
                            _flatten_chain(nodes, edges, node_id, child_id)
                            transformed = True

        else:
            # If the item does not contain a 'node' key but might have children
            if "children" in item and isinstance(item["children"], list):
                child_nodes, child_edges, current_id = extract_nodes_and_edges(
                    item["children"],
                    parent_id=parent_id,
                    start_id=current_id,
                    include_failed_pp=include_failed_pp,
                    deduplicate=deduplicate,
                )
                nodes.update(child_nodes)
                edges.extend(child_edges)

    return nodes, edges, current_id


def _flatten_chain(nodes, edges, parent_id, child_id):
    """
    Flatten a chain by removing 'child_id' node and connecting 'parent_id'
    directly to the child's children.
    Given a parent node that has a single child, this function removes the child
    node from the dictionary of nodes and reassigns the child's children to the parent.
    This is used for node deduplication and removing failed-pp nodes.

    Parameters
    ----------
    nodes : dict of {int : dict}
        A dictionary of node_id -> node_content.
    edges : list of tuple
        A list of tuples (parent_id, child_id, {}) representing edges in the infotree.
    parent_id : int
        The ID of the parent node.
    child_id : int
        The ID of the child node that should be removed.

    Returns
    -------
    None
        This function modifies the nodes and edges in place.
    """
    if child_id not in nodes:
        return

    # Remove the node from the dictionary
    del nodes[child_id]

    # Remove edge from parent_id -> child_id
    edges[:] = [e for e in edges if not (e[0] == parent_id and e[1] == child_id)]

    # Reassign child's children edges to the parent
    for i, (src, tgt, attr) in enumerate(edges):
        if src == child_id:
            edges[i] = (parent_id, tgt, attr)


def get_intervals(nodes):
    """
    Build a list of intervals from a given nodes dictionary.
    Each interval represents a tactic in the Lean file, capturing its
    start and finish positions, as well as the associated goals.

    Parameters
    ----------
    nodes : dict of {int : dict}
        A dictionary of node_id -> node_content.

    Returns
    -------
    intervals : list of dict
        A list of dictionaries, each containing:
          node_id, pp, start_line, start_col, finish_line, finish_col, goalsBefore, goalsAfter
    """
    intervals = []
    for node_id, node_content in nodes.items():
        stx_range = node_content.get("stx", {}).get("range", {})
        start_dict = stx_range.get("start", {})
        finish_dict = stx_range.get("finish", {})

        intervals.append(
            {
                "node_id": node_id,
                "pp": node_content.get("stx", {}).get("pp", ""),
                "start_line": start_dict.get("line", 0),
                "start_col": start_dict.get("column", 0),
                "finish_line": finish_dict.get("line", 0),
                "finish_col": finish_dict.get("column", 0),
                "goalsBefore": node_content.get("goalsBefore", []),
                "goalsAfter": node_content.get("goalsAfter", []),
            }
        )
    return intervals


def adjust_intervals(intervals):
    """
    Make intervals disjoint and create a file partition.
    Sort intervals by starting position, then set each interval's end to the next
    interval's start. This creates a sequence of adjacent intervals covering the file.

    Parameters
    ----------
    intervals : list of dict
        A list of dictionaries, each containing:
          node_id, pp, start_line, start_col, finish_line, finish_col, goalsBefore, goalsAfter

    Returns
    -------
    intervals_sorted : list of dict
        The updated intervals, sorted and trimmed so that they do not overlap.
    """
    intervals_sorted = sorted(
        intervals, key=lambda iv: (iv["start_line"], iv["start_col"])
    )
    for i in range(len(intervals_sorted) - 1):
        current = intervals_sorted[i]
        nxt = intervals_sorted[i + 1]
        current["finish_line"] = nxt["start_line"]
        current["finish_col"] = nxt["start_col"]
        current["goalsAfter"] = nxt["goalsBefore"]

    intervals_sorted = [
        iv
        for iv in intervals_sorted
        if not (
            iv["start_line"] == iv["finish_line"]
            and iv["start_col"] == iv["finish_col"]
        )
    ]

    return intervals_sorted


def retrieve_tactics(intervals, source_lines):
    """
    Extract tactic code snippets from source lines based on intervals.

    Parameters
    ----------
    intervals : list of dict
        A list of dictionaries, each containing:
          node_id, pp, start_line, start_col, finish_line, finish_col, goalsBefore, goalsAfter
        Note: At this point, the pp field does not exactly correspond to the positions.
    source_lines : list of str
        The lines of the Lean file, read into a list.

    Returns
    -------
    results : list of dict
        A list of intervals augmented with the 'tactic' text from the file.
        Each dict has keys: start, finish, goalsBefore, goalsAfter, tactic.
    """
    results = []
    for i in range(len(intervals)):
        iv = intervals[i]
        snippet_text = _extract_snippet(
            source_lines,
            iv["start_line"],
            iv["start_col"],
            iv["finish_line"],
            iv["finish_col"],
        )
        data = {
            # "start_line": iv["start_line"],
            # "start_col": iv["start_col"],
            # "finish_line": iv["finish_line"],
            # "finish_col": iv["finish_col"],
            # "start": (iv["start_line"], iv["start_col"]),
            # "finish": (iv["finish_line"], iv["finish_col"]),
            "goalsBefore": iv["goalsBefore"],
            "goalsAfter": iv["goalsAfter"],
            "tactic": snippet_text,
        }
        results.append(data)

    return results


def _extract_snippet(source_lines, start_line, start_col, finish_line, finish_col):
    """
    Extract a code snippet from the Lean source lines.

    Given a start and finish line-column pair, slice the lines to produce the exact text
    range in the Lean file. This handles both single-line and multi-line cases.

    Parameters
    ----------
    source_lines : list of str
        The lines read from the Lean file.
    start_line : int
        The 1-based starting line index.
    start_col : int
        The 0-based starting column index within start_line.
    finish_line : int
        The 1-based finishing line index.
    finish_col : int
        The 0-based finishing column index within finish_line.

    Returns
    -------
    str
        The extracted snippet from the Lean file, spanning (start_line, start_col)
        to (finish_line, finish_col).
    """
    # Single line case
    if start_line == finish_line:
        line_idx = start_line - 1
        line_text = source_lines[line_idx]
        return line_text[start_col:finish_col]

    # Multi-line case
    # 1) from start_col to end-of-line for start_line
    snippet_parts = []
    start_line_idx = start_line - 1
    line_text = source_lines[start_line_idx]
    snippet_parts.append(line_text[start_col:])

    # 2) full lines between (start_line+1) .. (finish_line-1)
    for line_idx in range(start_line_idx + 1, finish_line - 1):
        snippet_parts.append(source_lines[line_idx])

    # 3) from begin-of-line up to finish_col for finish_line
    last_line_idx = finish_line - 1
    last_line = source_lines[last_line_idx]
    snippet_parts.append(last_line[:finish_col])

    return "".join(snippet_parts)


def transfer_trailing_whitespace(intervals):
    """
    Transfer trailing whitespace from the end of one tactic to the beginning of the next one.

    Example:
      - interval[i]['tactic'] = "have h1 : ... := by\n    "
      - interval[i+1]['tactic'] = "apply mul_pos\n    "
      - ...

    After calling this function:
      - interval[i]['tactic']   = "have h1 : ... := by"
      - interval[i+1]['tactic'] = "\n    apply mul_pos"
      - ...

    Parameters
    ----------
    intervals : list of dict
        A list of dictionaries, each containing:
          start, finish, goalsBefore, goalsAfter, tactic.

    Returns
    -------
    None
        The 'tactic' fields in intervals are modified in place.
    """
    for i in range(len(intervals) - 1):
        current_tactic = intervals[i]["tactic"]
        next_tactic = intervals[i + 1]["tactic"]

        # Find the position of the last non-whitespace character
        j = len(current_tactic) - 1
        while j >= 0 and current_tactic[j] in (" ", "\t", "\n", "\r"):
            j -= 1

        # Everything from j+1 onward is trailing whitespace
        trailing_ws = current_tactic[j + 1 :]

        # Trim the trailing whitespace from the current tactic
        intervals[i]["tactic"] = current_tactic[: j + 1]

        # Prepend that whitespace to the next tactic
        intervals[i + 1]["tactic"] = trailing_ws + next_tactic


def check_intervals(intervals):
    """
    Check that the intervals have a valid start-finish ordering.
    This function checks if any interval's starting position is lexicographically
    after its finishing position, which indicates an invalid or inverted interval.

    Parameters
    ----------
    intervals : list of dict
        A list of dictionaries, each containing:
          start, finish, goalsBefore, goalsAfter, tactic.

    Returns
    -------
    None
        Prints an error message if an invalid interval is detected.
    """
    for i, iv in enumerate(intervals):
        if (iv["start_line"], iv["start_col"]) > (iv["finish_line"], iv["finish_col"]):
            print(f"ERROR: interval {i} has start after finish")
            # print(iv)


def merge_tactics(intervals):
    """
    Merge tactics that do not change the goals with their successor.

    Parameters
    ----------
    intervals : list of dict
        A list of dictionaries, each containing:
          start, finish, goalsBefore, goalsAfter, tactic.

    Returns
    -------
    merged_intervals : list of dict
        A list of dictionaries, each containing:
          start, finish, goalsBefore, goalsAfter, tactic.
        The tactics that do not change the goals are merged with their successor.
    """
    merged_intervals = []
    i = 0
    while i < len(intervals) - 1:
        current = intervals[i]
        next = intervals[i + 1]

        # Check if the goals are the same
        if current["goalsBefore"] == current["goalsAfter"]:
            # Merge the tactics
            merged_intervals.append(
                {
                    "goalsBefore": current["goalsBefore"],
                    "goalsAfter": next["goalsAfter"],
                    "tactic": current["tactic"] + next["tactic"],
                }
            )
            # Skip the next interval since it's already merged
            i += 1
        else:
            merged_intervals.append(current)

        i += 1

    # Add the last interval if it wasn't merged
    if i == len(intervals) - 1:
        merged_intervals.append(intervals[-1])

    return merged_intervals


def extract_data(infotree, source_code):
    """
    Performs the whole extraction process from an infotree and the corresponding Lean file.

    This function:
      - Extracts nodes and edges from the infotree,
      - Removes synthetic nodes,
      - Builds and adjusts intervals to partition the Lean file,
      - Retrieves tactics from the source file,
      - Transfers trailing whitespaces between consecutive tactics.

    Parameters
    ----------
    infotree : list of dict
        A list of dictionaries each containing 'node' and optionally 'children'.
    file_path : str
        The path to the Lean source file for retrieving text snippets.

    Returns
    -------
    intervals : list of dict
        A list of dictionaries, each containing:
          start, finish, goalsBefore, goalsAfter, tactic.
        The intervals are partitioned and cover the whole file.
    """
    # 1. Extract nodes and edges
    nodes, _, _ = extract_nodes_and_edges(
        infotree, include_failed_pp=False, deduplicate=True
    )

    # 2. Filter out the synthetic nodes
    nodes = {
        k: v
        for k, v in nodes.items()
        if not v.get("stx", {}).get("range", {}).get("synthetic", False)
    }

    # 3. Build raw intervals from nodes
    intervals = get_intervals(nodes)

    # 4. Adjust intervals so they become disjoint and partition the proof
    intervals = adjust_intervals(intervals)

    # 5. Load lines from the Lean file
    # with open(file_path, "r", encoding="utf-8") as file:
    #     source_lines = file.readlines()
    source_lines = source_code.split("\n")
    source_lines = [line + '\n' for line in source_lines[:-1]] + [source_lines[-1]]

    # 6. Extract the tactic for each final interval
    intervals = retrieve_tactics(intervals, source_lines)

    # 7. Transfer trailing whitespace
    transfer_trailing_whitespace(intervals)

    # 8. Merge tactics that do not change the goals to their successor
    intervals = merge_tactics(intervals)

    return intervals
