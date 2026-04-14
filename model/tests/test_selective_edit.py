import pytest

from scripts.selective_edit import (
    get_numbered_graphviz_code,
    extract_json_edit_plan,
    validate_actions,
    apply_graphviz_edit_command,
    apply_llm_edit_plan,
)


class TestGetNumberedGraphvizCode:
    def test_simple_directed_graph(self):
        code = "digraph G { A -> B; B -> C; }"
        numbered, parts = get_numbered_graphviz_code(code)

        assert len(parts) == 4
        assert parts[0] == "digraph G {"
        assert parts[1] == " A -> B;"
        assert parts[2] == " B -> C;"
        assert parts[3] == " }"
        assert numbered.startswith("1. digraph G {")
        assert "2.  A -> B;" in numbered

    def test_strips_embedded_newlines(self):
        code = "digraph G {\nA -> B;\nB -> C;\n}"
        _, parts = get_numbered_graphviz_code(code)

        for part in parts:
            assert "\n" not in part

    def test_skips_empty_segments(self):
        code = "A -> B;   ;B -> C;"
        _, parts = get_numbered_graphviz_code(code)

        assert all(part.strip() for part in parts)

    def test_empty_string(self):
        numbered, parts = get_numbered_graphviz_code("")

        assert parts == []
        assert numbered == ""

    def test_numbering_is_one_indexed(self):
        code = "A -> B; C -> D;"
        numbered, parts = get_numbered_graphviz_code(code)

        lines = numbered.split("\n")
        assert lines[0].startswith("1.")
        assert lines[-1].startswith(f"{len(parts)}.")


class TestExtractJsonEditPlan:
    def test_clean_json(self):
        text = '{"actions": []}'
        result = extract_json_edit_plan(text)

        assert result == {"actions": []}

    def test_json_embedded_in_prose(self):
        text = 'Here is the plan: {"actions": [{"command": "delete", "idx": 1}]} done.'
        result = extract_json_edit_plan(text)

        assert result == {"actions": [{"command": "delete", "idx": 1}]}

    def test_nested_braces(self):
        text = '{"actions": [{"command": "edit", "idx": 1, "content": "a"}]}'
        result = extract_json_edit_plan(text)

        assert len(result["actions"]) == 1
        assert result["actions"][0]["command"] == "edit"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="does not contain JSON"):
            extract_json_edit_plan("no json here")

    def test_unbalanced_braces_raises(self):
        with pytest.raises(ValueError, match="complete JSON object"):
            extract_json_edit_plan("{ unbalanced")

    def test_whitespace_only_clean_json(self):
        text = '   {"actions": []}   '
        result = extract_json_edit_plan(text)

        assert result == {"actions": []}


class TestValidateActions:
    def test_missing_actions_key(self):
        result = validate_actions({}, num_parts=5)
        assert result == []

    def test_actions_not_a_dict(self):
        result = validate_actions("not a dict", num_parts=5)
        assert result == []

    def test_actions_not_a_list(self):
        result = validate_actions({"actions": "not a list"}, num_parts=5)
        assert result == []

    def test_invalid_command_skipped(self):
        plan = {
            "actions": [
                {"command": "upsert", "idx": 1, "content": "x"},
                {"command": "delete", "idx": 1},
            ]
        }
        result = validate_actions(plan, num_parts=3)

        assert len(result) == 1
        assert result[0]["command"] == "delete"

    def test_action_not_a_dict_skipped(self):
        plan = {"actions": ["not a dict", {"command": "delete", "idx": 1}]}
        result = validate_actions(plan, num_parts=3)

        assert len(result) == 1

    def test_missing_idx_skipped(self):
        plan = {"actions": [{"command": "delete"}]}
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_non_integer_idx_skipped(self):
        plan = {"actions": [{"command": "delete", "idx": "1"}]}
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_add_idx_out_of_range(self):
        plan = {
            "actions": [
                {"command": "add", "idx": 0, "content": "x"},
                {"command": "add", "idx": 10, "content": "x"},
            ]
        }
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_add_idx_at_end_allowed(self):
        plan = {"actions": [{"command": "add", "idx": 4, "content": "x"}]}
        result = validate_actions(plan, num_parts=3)

        assert len(result) == 1
        assert result[0]["idx"] == 3

    def test_add_missing_content_skipped(self):
        plan = {"actions": [{"command": "add", "idx": 1}]}
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_add_empty_content_skipped(self):
        plan = {"actions": [{"command": "add", "idx": 1, "content": ""}]}
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_edit_idx_out_of_range(self):
        plan = {
            "actions": [
                {"command": "edit", "idx": 0, "content": "x"},
                {"command": "edit", "idx": 5, "content": "x"},
            ]
        }
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_edit_missing_content_skipped(self):
        plan = {"actions": [{"command": "edit", "idx": 1}]}
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_delete_idx_out_of_range(self):
        plan = {
            "actions": [
                {"command": "delete", "idx": 0},
                {"command": "delete", "idx": 5},
            ]
        }
        result = validate_actions(plan, num_parts=3)

        assert result == []

    def test_index_translation_to_zero_based(self):
        plan = {
            "actions": [
                {"command": "edit", "idx": 2, "content": "x"},
                {"command": "delete", "idx": 3},
            ]
        }
        result = validate_actions(plan, num_parts=3)

        assert result[0]["idx"] == 1
        assert result[1]["idx"] == 2

    def test_mixed_valid_and_invalid(self):
        plan = {
            "actions": [
                {"command": "edit", "idx": 1, "content": "valid"},
                {"command": "bogus", "idx": 1},
                {"command": "delete", "idx": 99},
                {"command": "add", "idx": 2, "content": "valid"},
            ]
        }
        result = validate_actions(plan, num_parts=3)

        assert len(result) == 2
        assert result[0]["command"] == "edit"
        assert result[1]["command"] == "add"


class TestApplyGraphvizEditCommand:
    def test_edit_only(self):
        parts = ["a;", "b;", "c;"]
        actions = [{"command": "edit", "idx": 1, "content": "B;"}]
        result = apply_graphviz_edit_command(parts, actions)

        assert result == "a;\nB;\nc;"

    def test_delete_only(self):
        parts = ["a;", "b;", "c;"]
        actions = [{"command": "delete", "idx": 1}]
        result = apply_graphviz_edit_command(parts, actions)

        assert result == "a;\nc;"

    def test_add_only(self):
        parts = ["a;", "c;"]
        actions = [{"command": "add", "idx": 1, "content": "b;"}]
        result = apply_graphviz_edit_command(parts, actions)

        assert result == "a;\nb;\nc;"

    def test_multiple_deletes_descending(self):
        parts = ["a;", "b;", "c;", "d;"]
        actions = [
            {"command": "delete", "idx": 1},
            {"command": "delete", "idx": 3},
        ]
        result = apply_graphviz_edit_command(parts, actions)

        assert result == "a;\nc;"

    def test_multiple_adds_ascending(self):
        parts = ["a;", "d;"]
        actions = [
            {"command": "add", "idx": 1, "content": "b;"},
            {"command": "add", "idx": 2, "content": "c;"},
        ]
        result = apply_graphviz_edit_command(parts, actions)

        assert result == "a;\nb;\nc;\nd;"

    def test_combined_edit_delete_add(self):
        parts = ["a;", "b;", "c;", "d;"]
        actions = [
            {"command": "edit", "idx": 0, "content": "A;"},
            {"command": "delete", "idx": 2},
            {"command": "add", "idx": 1, "content": "X;"},
        ]
        result = apply_graphviz_edit_command(parts, actions)

        assert result == "A;\nX;\nb;\nd;"

    def test_empty_actions_returns_joined_parts(self):
        parts = ["a;", "b;"]
        result = apply_graphviz_edit_command(parts, [])

        assert result == "a;\nb;"


class TestApplyLlmEditPlan:
    def test_full_pipeline(self):
        parts = ["digraph G {", "A -> B;", "B -> C;", "}"]
        llm_response = """
        {
            "actions": [
                {"command": "edit", "idx": 2, "content": "A -> X;"},
                {"command": "delete", "idx": 3}
            ]
        }
        """
        result = apply_llm_edit_plan(parts, llm_response)

        assert "A -> X;" in result
        assert "B -> C;" not in result
        assert "digraph G {" in result

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            apply_llm_edit_plan(["a;"], "not json at all")

    def test_invalid_actions_yield_unchanged_code(self):
        parts = ["a;", "b;"]
        llm_response = '{"actions": [{"command": "delete", "idx": 99}]}'
        result = apply_llm_edit_plan(parts, llm_response)

        assert result == "a;\nb;"
