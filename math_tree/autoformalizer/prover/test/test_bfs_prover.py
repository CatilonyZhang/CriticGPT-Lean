"""
usage: pytest prover/tests/test_bfs_prover.py
"""

import queue
from unittest.mock import MagicMock, patch

import pytest

from prover.bfs_prover import BestFirstSearchProver, InternalNode


# Utility functions to reduce redundancy
def setup_mock_node(codes, priority, out_edges=None):
    """Create a mock InternalNode."""
    node = MagicMock(spec=InternalNode)
    node.codes = codes
    node.priority = priority
    node.out_edges = out_edges or []
    return node


def initialize_prover_with_task(prover, task):
    """Set up the prover with a mock task."""
    mock_init_state = MagicMock()
    mock_init_state.is_valid = True
    prover.lean4_client.init_theorem.return_value = mock_init_state
    prover._intialize_search(task)
    assert prover.root is not None
    assert len(prover.nodes) == 1  # Root node


@pytest.fixture
def prover():
    """Fixture to set up a mock BestFirstSearchProver instance."""
    mock_tac_gen = MagicMock()
    mock_lean4_client = MagicMock()
    prover = BestFirstSearchProver(
        tac_gen=mock_tac_gen,
        lean4_client=mock_lean4_client,
        timeout=60,
        num_sampled_tactics=5,
    )
    prover.priority_queue = queue.PriorityQueue()
    return prover


def test_expand_and_update_priority_queue_empty(prover):
    """Test that the method handles an empty priority queue gracefully."""
    prover.priority_queue = MagicMock()
    prover.priority_queue.get.side_effect = queue.Empty

    with patch("prover.bfs_prover.logger.debug") as mock_logger:
        prover._expand_and_update()
        mock_logger.assert_called_with("Priority queue is empty")


def test_expand_and_update_valid_node(prover):
    """Test valid node expansion with tactics."""
    mock_task = {"formal_statement": "theorem example : ∀ x, x = x"}
    initialize_prover_with_task(prover, mock_task)

    mock_node = setup_mock_node(["mock_state"], 0.9)
    prover.priority_queue = MagicMock()
    prover.priority_queue.get.return_value = (-mock_node.priority, mock_node)

    def mock_construct_trees(states, parent_node):
        new_node = setup_mock_node(["new_state"], 0.5)
        prover.nodes.append(new_node)
        return [new_node]

    with patch.object(
        prover, "_generate_tactics", return_value=[[("tactic_1", -0.1)], "code"]
    ), patch.object(
        prover, "_run_seq_tactic", return_value=[([("new_state", -0.1)], True, "code")]
    ), patch.object(
        prover, "_construct_trees", side_effect=mock_construct_trees
    ):

        prover._expand_and_update()
        assert len(prover.nodes) == 2  # Root node + expanded node
        assert prover.nodes[1].codes == ["new_state"]
        assert prover.num_expansions == 1
        prover.priority_queue.task_done.assert_called_once()


def test_expand_and_update_invalid_tactic(prover):
    """Test expansion with an invalid tactic result."""
    mock_task = {"formal_statement": "theorem example : ∀ x, x = x"}
    initialize_prover_with_task(prover, mock_task)

    mock_node = setup_mock_node(["mock_state"], 0.9)
    prover.priority_queue = MagicMock()
    prover.priority_queue.get.return_value = (-mock_node.priority, mock_node)

    invalid_tactic_result = []  # No valid tactics
    with patch.object(
        prover, "_generate_tactics", return_value=[[("invalid_tactic", -0.1)], "code"]
    ), patch.object(
        prover, "_run_seq_tactic", return_value=invalid_tactic_result
    ), patch.object(
        prover, "_construct_trees"
    ) as mock_construct_trees:

        prover._expand_and_update()
        mock_construct_trees.assert_not_called()
        assert len(prover.nodes) == 1  # Only root node
        assert prover.num_expansions == 1
        prover.priority_queue.task_done.assert_called_once()


def test_expand_and_update_leaf_nodes(prover):
    """Test that leaf nodes are updated and their status is recomputed."""
    # Define a mock task for initialization
    mock_task = {"formal_statement": "theorem example : ∀ x, x = x"}
    initialize_prover_with_task(prover, mock_task)

    mock_node = setup_mock_node(["mock_state"], 0.9)
    prover.priority_queue = MagicMock()
    prover.priority_queue.get.return_value = (-mock_node.priority, mock_node)

    prover.priority_queue = MagicMock()
    prover.priority_queue.empty.return_value = False
    prover.priority_queue.get.return_value = (-mock_node.priority, mock_node)

    leaf_node = setup_mock_node(["leaf_state"], 0.5)
    prover.nodes.append(leaf_node)

    with patch.object(
        leaf_node, "_recompute_status"
    ) as mock_recompute_status, patch.object(
        leaf_node, "_recompute_distance_to_proof"
    ) as mock_recompute_distance, patch.object(
        prover, "_construct_trees", return_value=[leaf_node]
    ), patch.object(
        prover, "_run_seq_tactic", return_value=[([("leaf_state", -0.1)], True, "code")]
    ), patch.object(
        prover, "_generate_tactics", return_value=[[("tactic_1", -0.1)], "code"]
    ):

        prover._expand_and_update()

        # Assert that the leaf node was added to `nodes`
        assert len(prover.nodes) == 2  # Root node + 1 leaf node

        # Assert that `status` and `distance_to_proof` were recomputed for the leaf node
        mock_recompute_status.assert_called_once()
        mock_recompute_distance.assert_called_once()

        # Verify `num_expansions` was incremented
        assert prover.num_expansions == 1

        # Ensure the priority queue marked the task as done
        prover.priority_queue.task_done.assert_called_once()
