import re


class LeanFatalError(Exception):
    "Error raised when Lean process times out."
    pass


def parse_messages(messages):
    parsed_messages = []
    for msg in messages:
        severity = msg.get("severity", "info")
        data = msg.get("data", "")
        pos = msg.get("pos", {"line": 0, "column": 0})
        end_pos = msg.get("endPos", {"line": 0, "column": 0})
        parsed_messages.append(
            {"severity": severity, "message": data, "pos": pos, "endPos": end_pos}
        )
    return parsed_messages


def parse_error_message(message):
    match = re.match(r"^(.*?):\n(.*)", message, re.DOTALL)
    if match:
        severity = match.group(1)
        msg = match.group(2)
    else:
        severity = "error"
        msg = message
    return [
        {
            "severity": severity,
            "message": msg,
            "pos": {"line": 0, "column": 0},
            "endPos": {"line": 0, "column": 0},
        }
    ]


def parse_lean_response(response):
    messages = []
    if "messages" in response:
        messages = parse_messages(response.get("messages", []))
    elif "message" in response:
        messages = parse_error_message(response.get("message", ""))

    # TODO: @marco is it ok to filter out unsolved goals?
    # messages = list(filter(lambda x: "unsolved goals" not in x["message"], messages))
    # messages = sorted(messages, key=lambda x: (x["pos"]["line"], x["pos"]["column"]))

    line_num_to_message = {message["pos"]["line"]: message for message in messages}
    return line_num_to_message


def get_messages_for_lines(messages, start_line, end_line):
    selected_messages = []
    has_error = False
    is_unsolved_goals = False
    for idx in range(start_line, end_line + 1):
        if idx in messages:
            selected_messages.append(messages[idx])
            if messages[idx]["severity"] == "error":
                has_error = True
            if "unsolved goals" in messages[idx]["message"]:
                is_unsolved_goals = True
    return selected_messages, has_error, is_unsolved_goals


def reconstruct_proof(context, state):
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
                _ = current_state.statement
            break

    return {"context": context, "steps": proof_steps}


def get_indention(tactic):
    tactic = tactic.split("\n")
    while tactic[0].strip() == "":
        tactic = tactic[1:]
    tactic = "\n".join(tactic)
    return len(tactic) - len(tactic.lstrip())


def format_goals_with_indention(goals, indention):
    goals = goals.split("\n")
    goals = [indention + goal for goal in goals]
    return "\n".join(goals)


def split_proof_header(proof):
    proof = proof.strip()
    header_lines = []
    context_lines = []
    toggle = False
    proof_lines = proof.split("\n")
    index = 0
    for line in proof_lines:
        if line.startswith("import"):
            if toggle is False:
                toggle = True
            header_lines.append(line)
            index += 1
        else:
            if toggle is True:
                toggle = False
                break
    context_lines = proof_lines[index:]
    return "\n".join(header_lines).strip(), "\n".join(context_lines)


def split_logprobs(logprobs, tactics):
    def get_string(logprobs):
        strs = []
        for item in logprobs:
            strs.append(list(item.keys())[0])
        return "".join(strs)

    def get_logprob(logprobs):
        logprob = []
        for item in logprobs:
            logprob.append(list(item.values())[0])
        return sum(logprob)

    ret = []
    start_idx = 0
    end_idx = 0
    for idx, tactic in enumerate(tactics):
        tactic_text = tactic["tactic"]
        if "skip" in tactic_text and idx == len(tactics) - 1:
            if tactic_text.strip() == "skip":
                continue
            else:
                tactic_text = tactic_text[: tactic_text.index("skip")]
        if idx == 0:
            tactic_text = tactic_text[1:]
        tactic_text = tactic_text.strip()
        while tactic_text not in get_string(logprobs[start_idx:end_idx]):
            end_idx += 1
            if end_idx >= len(logprobs):
                return None
        ret.append(get_logprob(logprobs[start_idx:end_idx]))
        start_idx = end_idx
    return ret
