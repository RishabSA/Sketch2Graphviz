import networkx as nx
import pytest

from scripts.eval import (
    get_attribute_items,
    build_node_set,
    build_edge_set,
    build_node_attribute_set,
    build_edge_attribute_set,
    compute_precision_recall_f1,
    compute_networkx_graph_f1_metrics,
)


class TestGetAttributeItems:
    def test_all_keys_present(self):
        attributes = {"label": "A", "shape": "box", "color": "red"}
        result = get_attribute_items(attributes, ("label", "shape", "color"))

        assert result == (("color", "red"), ("label", "A"), ("shape", "box"))

    def test_some_keys_missing(self):
        attributes = {"label": "A"}
        result = get_attribute_items(attributes, ("label", "shape", "color"))

        assert result == (("label", "A"),)

    def test_no_keys_present(self):
        result = get_attribute_items({"foo": "bar"}, ("label", "shape"))
        assert result == ()

    def test_sorted_output(self):
        attributes = {"shape": "box", "color": "red", "label": "A"}
        result = get_attribute_items(attributes, ("shape", "color", "label"))

        keys = [item[0] for item in result]
        assert keys == sorted(keys)


class TestBuildNodeSet:
    def test_directed_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B", "C"])

        assert build_node_set(graph) == {"A", "B", "C"}

    def test_undirected_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(["X", "Y"])

        assert build_node_set(graph) == {"X", "Y"}

    def test_isolated_nodes_included(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B"])
        graph.add_edge("A", "B")
        graph.add_node("C")

        assert build_node_set(graph) == {"A", "B", "C"}

    def test_empty_graph(self):
        assert build_node_set(nx.DiGraph()) == set()


class TestBuildEdgeSet:
    def test_directed_preserves_order(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "A")])

        assert build_edge_set(graph) == {("A", "B"), ("B", "A")}

    def test_undirected_normalizes_order(self):
        graph = nx.Graph()
        graph.add_edge("B", "A")

        result = build_edge_set(graph)
        assert result == {("A", "B")}

    def test_self_loop(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "A")

        assert build_edge_set(graph) == {("A", "A")}

    def test_undirected_dedupes_reciprocal(self):
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "A")])

        assert build_edge_set(graph) == {("A", "B")}


class TestBuildNodeAttributeSet:
    def test_node_with_attributes(self):
        graph = nx.DiGraph()
        graph.add_node("A", label="Alpha", shape="box")
        graph.add_node("B", label="Beta", shape="ellipse")

        result = build_node_attribute_set(graph, ("label", "shape"))

        assert ("A", (("label", "Alpha"), ("shape", "box"))) in result
        assert ("B", (("label", "Beta"), ("shape", "ellipse"))) in result

    def test_node_without_matching_keys(self):
        graph = nx.DiGraph()
        graph.add_node("A", color="red")

        result = build_node_attribute_set(graph, ("label", "shape"))
        assert ("A", ()) in result

    def test_partial_attribute_match(self):
        graph = nx.DiGraph()
        graph.add_node("A", label="Alpha")

        result = build_node_attribute_set(graph, ("label", "shape"))
        assert ("A", (("label", "Alpha"),)) in result


class TestBuildEdgeAttributeSet:
    def test_directed_simple_graph(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", label="flows")

        result = build_edge_attribute_set(graph, ("label",))
        assert ("A", "B", (("label", "flows"),)) in result

    def test_undirected_normalizes_endpoints(self):
        graph = nx.Graph()
        graph.add_edge("B", "A", label="connects")

        result = build_edge_attribute_set(graph, ("label",))
        assert ("A", "B", (("label", "connects"),)) in result

    def test_multigraph(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", label="first")
        graph.add_edge("A", "B", label="second")

        result = build_edge_attribute_set(graph, ("label",))
        assert ("A", "B", (("label", "first"),)) in result
        assert ("A", "B", (("label", "second"),)) in result

    def test_edge_without_matching_keys(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", color="blue")

        result = build_edge_attribute_set(graph, ("label",))
        assert ("A", "B", ()) in result


class TestComputePrecisionRecallF1:
    def test_perfect_match(self):
        result = compute_precision_recall_f1({"a", "b"}, {"a", "b"})

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_overlap(self):
        result = compute_precision_recall_f1({"a", "b", "c"}, {"a", "b", "d"})

        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["f1"] == pytest.approx(2 / 3)

    def test_subset_predicted(self):
        result = compute_precision_recall_f1({"a"}, {"a", "b"})

        assert result["precision"] == 1.0
        assert result["recall"] == 0.5
        assert result["f1"] == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))

    def test_superset_predicted(self):
        result = compute_precision_recall_f1({"a", "b", "c"}, {"a"})

        assert result["precision"] == pytest.approx(1 / 3)
        assert result["recall"] == 1.0

    def test_no_overlap_raises_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            compute_precision_recall_f1({"a"}, {"b"})


class TestComputeNetworkxGraphF1Metrics:
    def test_identical_graphs(self):
        graph = nx.DiGraph()
        graph.add_node("A", label="Alpha", shape="box")
        graph.add_node("B", label="Beta", shape="ellipse")
        graph.add_edge("A", "B", label="flows", style="solid")

        result = compute_networkx_graph_f1_metrics(graph, graph)

        assert result["node_f1"] == 1.0
        assert result["edge_f1"] == 1.0
        assert result["node_attribute_f1"] == 1.0
        assert result["edge_attribute_f1"] == 1.0

    def test_extra_predicted_node_reduces_precision(self):
        original = nx.DiGraph()
        original.add_nodes_from(["A", "B"])
        original.add_edge("A", "B")

        generated = nx.DiGraph()
        generated.add_nodes_from(["A", "B", "C"])
        generated.add_edge("A", "B")

        result = compute_networkx_graph_f1_metrics(original, generated)

        assert result["node_recall"] == 1.0
        assert result["node_precision"] == pytest.approx(2 / 3)

    def test_same_structure_one_wrong_attribute(self):
        original = nx.DiGraph()
        original.add_node("A", label="Alpha")
        original.add_node("B", label="Beta")
        original.add_edge("A", "B")

        generated = nx.DiGraph()
        generated.add_node("A", label="Alpha")
        generated.add_node("B", label="WrongB")
        generated.add_edge("A", "B")

        result = compute_networkx_graph_f1_metrics(original, generated)

        assert result["node_f1"] == 1.0
        assert result["edge_f1"] == 1.0
        assert result["node_attribute_f1"] == pytest.approx(0.5)

    def test_partially_overlapping_graphs(self):
        original = nx.DiGraph()
        original.add_edges_from([("A", "B"), ("B", "C")])

        generated = nx.DiGraph()
        generated.add_edges_from([("A", "B"), ("B", "D")])

        result = compute_networkx_graph_f1_metrics(original, generated)

        assert result["node_precision"] == pytest.approx(2 / 3)
        assert result["node_recall"] == pytest.approx(2 / 3)
        assert result["edge_precision"] == 0.5
        assert result["edge_recall"] == 0.5
