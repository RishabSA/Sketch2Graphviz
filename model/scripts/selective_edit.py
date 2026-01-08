import re
import json
import logging


def get_numbered_graphviz_code(graphviz_code: str) -> tuple[str, list[str]]:
    separators = r"(?<=[{};])"
    graphviz_parts = re.split(separators, graphviz_code)

    cleaned_graphviz_parts = [p.replace("\n", "") for p in graphviz_parts if p.strip()]

    numbered_graphviz_code = "\n".join(
        [f"{i + 1}. {part}" for i, part in enumerate(cleaned_graphviz_parts)]
    )

    return numbered_graphviz_code, cleaned_graphviz_parts


def apply_graphviz_edit_command(graphviz_parts: list[str], actions: list[dict]) -> str:
    edits = [action for action in actions if action["command"] == "edit"]
    deletes = [action for action in actions if action["command"] == "delete"]
    adds = [action for action in actions if action["command"] == "add"]

    # Edit
    for action in edits:
        graphviz_parts[action["idx"]] = action["content"]

    # Delete in descending order
    for action in sorted(deletes, key=lambda x: x["idx"], reverse=True):
        del graphviz_parts[action["idx"]]

    # Add in ascending order
    for action in sorted(adds, key=lambda x: x["idx"]):
        graphviz_parts.insert(action["idx"], action["content"])

    return "\n".join(graphviz_parts)


def extract_json_edit_plan(text: str) -> dict[str, any]:
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        return json.loads(text_stripped)

    start = text.find("{")
    if start == -1:
        raise ValueError("The provided response does not contain JSON.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]

                return json.loads(candidate)

    raise ValueError("Could not find a complete JSON object in the response.")


def validate_actions(
    plan: dict[str, any],
    num_parts: int,
) -> list[dict[str, any]]:
    if not isinstance(plan, dict) or "actions" not in plan:
        logging.warning("JSON response does not contain the actions key.")
        return []

    actions = plan["actions"]
    if not isinstance(actions, list):
        logging.warning("actions must be a list.")
        return []

    validated_actions = []

    for action in actions:
        if not isinstance(action, dict):
            logging.warning("Each action must be a JSON object.")
            continue

        command = action.get("command")
        if command not in ["add", "edit", "delete"]:
            logging.warning(f"Invalid command: {command}")
            continue

        if "idx" not in action or not isinstance(action["idx"], int):
            logging.warning("Each action must include integer 'idx'.")
            continue

        idx = action["idx"]

        if command == "add":
            if idx < 1 or idx > (num_parts + 1):
                logging.warning(
                    f"Add idx out of range: {idx} (valid 1 - {num_parts + 1})"
                )
                continue
            if (
                "content" not in action
                or not isinstance(action["content"], str)
                or not action["content"]
            ):
                logging.warning("Add requires content.")
                continue

            validated_actions.append(
                {"command": "add", "idx": idx - 1, "content": action["content"]}
            )
        elif command == "edit":
            if idx < 1 or idx > num_parts:
                logging.warning(f"Edit idx out of range: {idx} (valid 1 - {num_parts})")
                continue
            if (
                "content" not in action
                or not isinstance(action["content"], str)
                or not action["content"]
            ):
                logging.warning("Edit requires content.")
                continue

            validated_actions.append(
                {"command": "edit", "idx": idx - 1, "content": action["content"]}
            )
        elif command == "delete":
            if idx < 1 or idx > num_parts:
                logging.warning(
                    f"Delete idx out of range: {idx} (valid 1 - {num_parts})"
                )
                continue

            validated_actions.append({"command": "delete", "idx": idx - 1})

    return validated_actions


def apply_llm_edit_plan(graphviz_parts: list[str], llm_response_text: str) -> str:
    json_edit_plan = extract_json_edit_plan(llm_response_text)
    validated_actions = validate_actions(json_edit_plan, num_parts=len(graphviz_parts))

    return apply_graphviz_edit_command(graphviz_parts, validated_actions)


if __name__ == "__main__":
    base_code = """
digraph G7 {
    rankdir=TB;
    s7_root [label=< <B>Root</B><BR/>R>];
    s7_l [label=< <B>Left</B><BR/>L>];
    s7_r [label=< <B>Right</B><BR/>R>];
    s7_ll [label=< <B>LL</B><BR/>L1>];
    s7_lr [label=< <B>LR</B><BR/>L2>];
    s7_root -> s7_l;
    s7_root -> s7_r;
    s7_l -> s7_ll;
    s7_l -> s7_lr;
}
    """

    numbered_graphviz_code, graphviz_parts = get_numbered_graphviz_code(base_code)
    print(f"Numbered Graphviz Code:\n{numbered_graphviz_code}\n")

    example_llm_response = """
{
    "actions": [
        {"command": "add", "idx": 4, "content": "s7_new [label=\\"New\\"];"},
        {"command": "edit", "idx": 2, "content": "rankdir=LR;"},
        {"command": "delete", "idx": 12}
    ]
}
"""

    updated_graphviz_code = apply_llm_edit_plan(
        graphviz_parts=graphviz_parts, llm_response_text=example_llm_response
    )

    print(f"Updated Graphviz code:\n{updated_graphviz_code}\n")
