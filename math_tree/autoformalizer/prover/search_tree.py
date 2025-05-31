"""Definitions of the search tree used by the prover.
This implementation is based on the code in lean-dojo/ReProver.
"""

import json
import math
import pathlib
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Dict, List, Optional

from autoformalizer.repl_lean_feedback.state import State


class LeanEvent(Enum):
    """Events that can be emitted by the Lean prover."""

    PROOF_FINISHED = "ProofFinished"
    ERROR = "Error"
    TIMEOUT = "Timeout"
    PROOF_GIVEN_UP = "ProofGivenUp"


class Status(Enum):
    """Status of a node or a proof search."""

    PROVED = "Proved"  # This node (or search) has at least one known proof.
    FAILED = "Failed"  # This node (or search) has exhausted its options and cannot be proved within the current run.
    OPEN = "Open"  # This node (or search) has not been proven or given up on yet.
    SYSTEM_ERROR = (
        "SystemError"  # This node (or search) has not been proven or given up on yet.
    )
    INIT_FAILED = (
        "InitFailed"  # This node (or search) has not been proven or given up on yet.
    )


@dataclass
class Node(ABC):

    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)

    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError

    @property
    @abstractmethod
    def distance_to_proof(self) -> int:
        "The smallest number of steps to a proof."
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize this node to a dictionary.
        id_map maps node objects to unique IDs so we can reference them consistently.
        """
        raise NotImplementedError


@dataclass
class ProofFinishedNode(Node):
    state: "State" = field(compare=True)
    inner: LeanEvent = LeanEvent.PROOF_FINISHED
    status = Status.PROVED
    distance_to_proof = 0
    is_terminal = True
    codes: List["State"] = field(compare=False, default_factory=list)

    def __hash__(self):
        return hash(str(self.state.goals))

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "id": self.id,
            "type": "ProofFinishedNode",
            "state_id": self.state.id,
            "inner": self.inner.value,
            "status": self.status.value,
            "distance_to_proof": self.distance_to_proof,
            "is_terminal": self.is_terminal,
            "code_ids": [s.id for s in self.codes],
        }

    @classmethod
    def from_dict(cls, data: dict, states_map: Dict[str, State]) -> "ProofFinishedNode":
        """
        Reconstruct a ProofFinishedNode from a dict.
        The states_map is used to look up State objects by their ID.
        """
        node = cls(
            state=states_map[data["state_id"]],  # fetch the actual State by ID
            codes=[states_map[cid] for cid in data["code_ids"]],
        )
        node.id = data["id"]
        node.inner = LeanEvent(data["inner"])  # e.g., "ProofFinished"
        node.distance_to_proof = data["distance_to_proof"]
        node.is_terminal = data["is_terminal"]
        # status is always PROVED by design, but you can reassign if you want
        # node.status = Status(data["status"])
        return node


@dataclass
class ErrorNode(Node):
    state: "State" = field(compare=True)
    inner: LeanEvent = LeanEvent.ERROR
    status = Status.FAILED
    distance_to_proof = math.inf
    is_terminal = True
    codes: List["State"] = field(compare=False, default_factory=list)

    def __hash__(self):
        return hash(str(self.state.goals))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "ErrorNode",
            "state_id": self.state.id,
            "inner": self.inner.value,
            "status": self.status.value,
            "distance_to_proof": self.distance_to_proof,
            "is_terminal": self.is_terminal,
            "code_ids": [s.id for s in self.codes],
        }

    @classmethod
    def from_dict(cls, data: dict, states_map: Dict[str, State]) -> "ErrorNode":
        node = cls(
            state=states_map[data["state_id"]],
            codes=[states_map[cid] for cid in data["code_ids"]],
        )
        node.id = data["id"]
        node.inner = LeanEvent(data["inner"])
        # node.status = Status(data["status"])  # usually "Failed"
        node.distance_to_proof = data["distance_to_proof"]
        node.is_terminal = data["is_terminal"]
        return node


@total_ordering
@dataclass(unsafe_hash=True)
class InternalNode(Node):
    """
    An internal node in the search tree, representing a nonterminal state.

    Nodes are sorted by _inverse_ priority, for compatibility with the `heapq` library.
    That is, node_a < node_b is true if node_a has _higher_ priority than node_b.
    """

    # Goal state this node represents. Two nodes are considered equal if their states
    # are equal; this is the only hashed field and must not be changed.
    state: "State" = field(compare=True)

    # The sum of action logprobs along edges from the root to this node
    cumulative_logprob: float = field(compare=False, repr=False, default=-1e9)

    # All edges known to lead to this node.
    # May change at any time as other nodes are explored.
    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    # All edges out of this node that we've considered, or None for unexplored nodes.
    # When a node is explored, this list is populated, and must not change after that.
    out_edges: Optional[List["Edge"]] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    # A node is proved if any child is proved, and failed if every child is failed
    # (or there are no children). A node that is proved or failed cannot change status
    # because nothing is ever added to out_edges. _status is recomputed on an as-needed
    # basis by children, since proving or failing a child may prove or fail this node.
    _status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)

    is_terminal = False  # type: ignore[override]
    is_explored = False  # type: ignore[override]

    # Number of steps separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    codes: List["State"] = field(compare=False, default_factory=list)
    visit_count: int = field(default=0, init=False, compare=False, repr=True)
    value: float = field(default=0, init=False, compare=False, repr=True)

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, s: Status):
        self._status = s

    def _recompute_status(self):
        """
        Recursively update the status of the current node and its ancestors.
        """
        assert self.out_edges is not None

        # If this node is proved or failed, nothing can change that
        if self._status != Status.OPEN:
            return

        # If any child is proved, this node is proved, and so are parents recursively
        if any(edge.dst.status == Status.PROVED for edge in self.out_edges):
            self._status = Status.PROVED

        # If this node was proved or failed, parents may need to recompute.
        # This is guaranteed to terminate because only open nodes can change, and
        # there are a finite number of open nodes in the tree.
        if self._status != Status.OPEN:
            for edge in self.in_edges:
                edge.src._recompute_status()

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    def _recompute_distance_to_proof(self):
        """
        Recursively update the distance_to_proof of the current node and its ancestors.
        """
        if self.out_edges:
            distance = min(edge.distance_to_proof() for edge in self.out_edges)
        else:
            distance = math.inf

        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            for edge in self.in_edges:
                edge.src._recompute_distance_to_proof()

    # NOTE: Nodes are compared by _negative_ priority, to make heapq act as a max-priority-queue.
    @property
    def priority(self) -> float:
        return self.cumulative_logprob

    def __lt__(self, other: "InternalNode") -> bool:
        return self.priority > other.priority

    def extract_proof(self) -> Optional[List["Edge"]]:
        """
        Extract a proof of the current node as a sequence of edges.
        """
        if self.status != Status.PROVED:
            return None
        # assert self.is_explored

        proving_edge = min(
            self.out_edges,
            key=Edge.distance_to_proof,
        )

        if proving_edge.dst.is_terminal:
            # Base case: this edge is all that's required to finish the proof
            assert isinstance(proving_edge.dst, ProofFinishedNode)
            return random.choice(proving_edge.dst.codes).tactics
        else:
            # Recursive case: prove the child, then add this edge
            assert isinstance(proving_edge.dst, InternalNode)
            return proving_edge.dst.extract_proof()

    def update_reward(self, reward: float, gamma: float = 0.99):
        self.reward = self.reward * gamma + reward
        self.visit_count = self.visit_count * gamma + 1.0
        self.value = self.reward / max(self.visit_count, 1e-2)

    def to_dict(self) -> dict:
        """Serialize this node into a dict, referencing edges by ID."""
        return {
            "id": self.id,
            "type": "InternalNode",
            "state_id": self.state.id,
            "cumulative_logprob": self.cumulative_logprob,
            "status": self._status.value,
            "distance_to_proof": self._distance_to_proof,
            "is_terminal": self.is_terminal,
            "is_explored": self.is_explored,
            "code_ids": [s.id for s in self.codes],
            "visit_count": self.visit_count,
            "value": self.value,
            "in_edge_ids": [edge.id for edge in self.in_edges],
            "out_edge_ids": (
                [edge.id for edge in self.out_edges] if self.out_edges else []
            ),
        }

    @classmethod
    def from_dict(cls, data: dict, states_map: Dict[str, State]) -> "InternalNode":
        node = cls(
            state=states_map[data["state_id"]],
            cumulative_logprob=data["cumulative_logprob"],
            codes=[states_map[cid] for cid in data["code_ids"]],
        )
        node.id = data["id"]
        node.is_terminal = data["is_terminal"]
        node.is_explored = data["is_explored"]
        node.visit_count = data["visit_count"]
        node.value = data["value"]
        node._status = Status(data["status"])
        node._distance_to_proof = data["distance_to_proof"]
        return node

    #########
    # Debug #
    #########

    def check_invariants(self):
        """
        Perform some sanity checks.
        """
        if not self.is_explored:
            assert self.status == Status.OPEN
            return  # Nothing more can be said about unexplored nodes

        for edge in self.in_edges:
            assert edge.dst is self

        if self.out_edges == []:
            assert self.status == Status.FAILED
        else:
            for edge in self.out_edges:  # type: ignore
                assert edge.src is self

        if self.status == Status.PROVED:
            assert self.out_edges
            assert any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert all(edge.dst.status == Status.PROVED for edge in self.in_edges)

            proof_by_steps = self.extract_proof()
            assert proof_by_steps is not None
            assert self.distance_to_proof == len(proof_by_steps)

        elif self.status == Status.FAILED:
            assert self.out_edges is not None
            assert all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.extract_proof() is None
        elif self.status == Status.OPEN:
            assert self.out_edges
            assert not any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert not all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.extract_proof() is None


@dataclass
class Edge:
    """An edge in the search tree, representing a tactic."""

    # tactic: str = field(compare=True)
    src: InternalNode = field(repr=False, compare=True)
    dst: Node = field(repr=False, compare=True)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __hash__(self):
        hash_str = str(self.src.state.goals) + str(self.dst.state.goals)
        return hash(hash_str)

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof

    def to_dict(self) -> dict:
        """Serialize this edge into a dict, referencing nodes by ID."""
        return {
            "id": self.id,
            "src_id": self.src.id,
            "dst_id": self.dst.id,
        }

    @classmethod
    def from_dict(cls, data: dict, id_to_node: Dict[str, Node]) -> "Edge":
        """
        Build an Edge from a dict.
        We rely on a map of node_id -> Node that must already be created.
        """
        src_node = id_to_node[data["src_id"]]
        dst_node = id_to_node[data["dst_id"]]
        edge = cls(src=src_node, dst=dst_node)
        edge.id = data["id"]
        return edge


def serialize_tree(root: Node, file_path: pathlib.Path) -> dict:
    """
    Traverse all Nodes and Edges reachable from 'root', and return
    a JSON-friendly dictionary of the entire subgraph.
    """
    # node_id -> node.to_dict()
    node_dicts: Dict[str, dict] = {}
    # edge_id -> edge.to_dict()
    edge_dicts: Dict[str, dict] = {}

    visited_nodes = set()
    queue = [root]

    while queue:
        current = queue.pop()
        if current in visited_nodes:
            continue
        visited_nodes.add(current)

        # Serialize the current node
        node_json = current.to_dict()
        node_dicts[current.id] = node_json

        # If it's an InternalNode, explore out_edges
        if isinstance(current, InternalNode) and current.out_edges:
            for edge in current.out_edges:
                # Serialize the edge
                edge_dicts[edge.id] = edge.to_dict()
                # Enqueue the destination node
                if edge.dst not in visited_nodes:
                    queue.append(edge.dst)

    # Return a top-level dict with root ID and all nodes/edges
    serialized_tree = {
        "root_id": root.id,
        "nodes": node_dicts,
        "edges": edge_dicts,
    }

    with open(file_path, "w") as f:
        json.dump(serialized_tree, f)


def deserialize_tree(file_path: pathlib.Path, states_map: Dict[str, State]):
    """
    Reconstructs a tree of Nodes (and their Edges) from `data`, using per-class from_dict methods.
    The `states_map` provides State objects keyed by their ID.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    node_data = data["nodes"]  # { node_id -> dict describing the node }
    edge_data = data["edges"]  # { edge_id -> dict describing the edge }
    root_id = data["root_id"]

    # 1) Create node shells by calling the appropriate from_dict for each node type
    id_to_node: Dict[str, Node] = {}
    for node_id, n_dict in node_data.items():
        node_type = n_dict["type"]
        if node_type == "ProofFinishedNode":
            node = ProofFinishedNode.from_dict(n_dict, states_map)
        elif node_type == "ErrorNode":
            node = ErrorNode.from_dict(n_dict, states_map)
        elif node_type == "InternalNode":
            node = InternalNode.from_dict(n_dict, states_map)
        else:
            raise ValueError(f"Unknown node type: {node_type}")
        id_to_node[node_id] = node

    # 2) Create edge objects and connect them to src/dst nodes
    id_to_edge: Dict[str, Edge] = {}
    for edge_id, e_dict in edge_data.items():
        edge = Edge.from_dict(e_dict, id_to_node)
        id_to_edge[edge_id] = edge

    # 3) Populate each node's in_edges and out_edges
    #    (only InternalNodes have them).
    for edge_id, e_dict in edge_data.items():
        edge = id_to_edge[edge_id]
        src_node = edge.src
        dst_node = edge.dst

        if isinstance(src_node, InternalNode):
            src_node.out_edges.append(edge)
        if isinstance(dst_node, InternalNode):
            dst_node.in_edges.append(edge)

    # 4) Return the root node
    return id_to_node[root_id], id_to_node
