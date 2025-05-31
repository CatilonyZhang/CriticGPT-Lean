import json
import math
import pathlib
import queue
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

from autoformalizer.clients.lean4_client import Lean4Client
from autoformalizer.eval_utils.lean_feedback import has_error
from autoformalizer.repl_lean_feedback.repl_utils import (
    LeanFatalError,
    format_goals_with_indention,
    get_indention,
    reconstruct_proof,
)
from autoformalizer.repl_lean_feedback.state import (
    State,
    deserialize_states,
    serialize_states,
)
from prover.search_tree import (
    Edge,
    ErrorNode,
    InternalNode,
    ProofFinishedNode,
    Status,
    deserialize_tree,
    serialize_tree,
)

from .tactic_generator import TacticGenerator


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: str
    status: Status
    proof: Optional[List[str]]

    # Some statistics during proof search.
    actor_time: float = 0.0
    environment_time: float = 0.0
    total_time: float = 0.0
    num_total_nodes: int = 0
    num_searched_nodes: int = 0

    def to_dict(self):
        return {
            "theorem": self.theorem,
            "status": self.status.value,
            "proof": self.proof,
            "actor_time": self.actor_time,
            "environment_time": self.environment_time,
            "total_time": self.total_time,
            "num_total_nodes": self.num_total_nodes,
            "num_searched_nodes": self.num_searched_nodes,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            theorem=data["theorem"],
            status=Status(data["status"]),
            proof=data["proof"],
            actor_time=data["actor_time"],
            environment_time=data["environment_time"],
            total_time=data["total_time"],
            num_total_nodes=data["num_total_nodes"],
            num_searched_nodes=data["num_searched_nodes"],
        )


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen: "TacticGenerator",  # A given tactic generator.
        lean4_client: "Lean4Client",
        timeout: int,
        num_sampled_tactics: int,
        max_expansions: Optional[int] = None,
        prompt_type: str = "text",
        temperature: float = 1.0,
        max_length: int = 2048,
        debug: bool = False,
        checkpoint_dir: pathlib.Path = None,
        serialize_interval: int = 120,
        resume_from_checkpoint: bool = False,
    ) -> None:
        """
        Arguments:
            - tac_gen (TacticGenerator): A given tactic generator, subclass of TacticGenerator.
            - lean4_client (Lean4Client): A given Lean4 client.
            - timeout (int): The timeout for the proof search with lean4 client.
            - num_sampled_tactics (int): The number of tactics to sample from the model.
            - max_expansions (int): The maximum number of nodes to expand in the whole search process.
            - prompt_type (str): The type of prompt to use, ['text', 'chat'].
            - temperature (float): The temperature for the language model.
            - max_length (int): The maximum length of the proof (in tokens).
            - debug (bool): Whether to run in debug mode (more log and checks).
        """
        self.tac_gen = tac_gen
        self.lean4_client = lean4_client
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.max_expansions = max_expansions
        self.prompt_type = prompt_type
        self.temperature = temperature
        self.max_length = max_length
        self.debug = debug
        self.total_time = None
        self.checkpoint_dir = checkpoint_dir
        self.serialize_interval = serialize_interval
        self.last_serialized = time.time()
        self.serialized_states = set()
        self.resume_from_checkpoint = resume_from_checkpoint

    def _intialize_search(self, task: str) -> None:
        """Initialize the search for a given theorem."""
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0
        self.is_valid_no_sorry = False
        theorem = task.get("formal_statement")

        # enforce the tactic mode:
        if theorem.rstrip().endswith(":="):
            theorem = theorem.rstrip() + " by"

        init_state = self.lean4_client.init_theorem(theorem, 60)
        logger.debug(f"Initial state: {init_state}")
        if init_state is None or not init_state.is_valid:
            logger.warning(f"Invalid initial state for {theorem}")
            return False

        self.root = InternalNode(
            state=init_state,
            cumulative_logprob=0.0,
        )
        self.root.codes.append(init_state)
        self.nodes = [self.root]
        self.states = [init_state]
        return True

    def _load_checkpoint(self) -> bool:
        """Load the state and tree from a checkpoint."""
        logger.debug("Attempting to load checkpoint...")
        time_elapsed = time.time()
        try:
            meta_data_file = self.checkpoint_dir / "meta_data.json"
            state_file = self.checkpoint_dir / "state.json"
            tree_file = self.checkpoint_dir / "tree.json"

            if meta_data_file.exists() and state_file.exists() and tree_file.exists():
                with open(meta_data_file, "r") as f:
                    meta_data = json.load(f)
                self.actor_time = meta_data["actor_time"]
                self.environment_time = meta_data["environment_time"]
                self.num_expansions = meta_data["num_expansions"]
                self.is_valid_no_sorry = meta_data["is_valid_no_sorry"]

                # Load the states and tree
                states_maps = deserialize_states(state_file)
                self.states = list(states_maps.values())
                self.serialized_states = set(states_maps.keys())

                self.root, nodes_map = deserialize_tree(tree_file, states_maps)
                self.nodes = list(nodes_map.values())

                self.priority_queue = queue.PriorityQueue()
                for logp, node_id in meta_data["priority_queue"]:
                    node = nodes_map[node_id]
                    self.priority_queue.put((-logp, node))
            else:
                logger.warning("Checkpoint does not exist.")
                return False

        except Exception as e:
            logger.error(f"Failed to load checkpoint {e}")
            logger.debug(
                f"Failed load checkpoint in {time.time() - time_elapsed:.2f} seconds."
            )
            return False
        logger.debug(f"Loaded checkpoint in {time.time() - time_elapsed:.2f} seconds.")
        return True

    def _save_checkpoint(self, search_result=None) -> bool:
        """Serialize the current state and tree to files."""
        logger.debug("Attempting to save checkpoint...")
        time_elapsed = time.time()
        try:
            meta_data_file = self.checkpoint_dir / "meta_data.json"
            state_file = self.checkpoint_dir / "state.json"
            tree_file = self.checkpoint_dir / "tree.json"

            priority_queue = [
                (logp, node.id) for logp, node in list(self.priority_queue.queue)
            ]

            meta_data = {
                "actor_time": self.actor_time,
                "environment_time": self.environment_time,
                "num_expansions": self.num_expansions,
                "is_valid_no_sorry": self.is_valid_no_sorry,
                "search_result": search_result.to_dict() if search_result else None,
                "priority_queue": priority_queue,
            }
            with open(meta_data_file, "w") as f:
                json.dump(meta_data, f)

            self.serialized_states = serialize_states(
                self.states, state_file, self.serialized_states
            )
            serialize_tree(self.root, tree_file)
        except Exception as e:
            logger.error(f"Failed to serialize checkpoint: {e}")
            logger.debug(
                f"Failed save checkpoint in {time.time() - time_elapsed:.2f} seconds."
            )
            return False
        logger.debug(f"Saved checkpoint in {time.time() - time_elapsed:.2f} seconds.")
        return True

    def search(self, task: dict) -> Optional[SearchResult]:
        """
        Overview:
            The main function to search for proofs.
        Arguments:
            - task (dict): A given task, which contains the theorem to prove and the proof header, \
                "formal_statement" and "header" respectively.
        Returns:
            - Optional[SearchResult]: The result of the proof search, which contains the theorem, \
                the status of the proof, and the proof if the proof is found.
        """
        logger.info(f"Proving {[task]}")
        self.theorem = task.get("formal_statement")

        if not self.resume_from_checkpoint:
            init_success = self._intialize_search(task)
        else:
            # try loading the checkpoint
            init_success = self._load_checkpoint()
            if not init_success:
                init_success = self._intialize_search(task)

        if not init_success:
            return SearchResult(
                theorem=self.theorem,
                status=Status.INIT_FAILED,
                proof=None,
            )

        # step 2: search for the proof
        try:
            self._search()
        except LeanFatalError as e:
            logger.error(f"Proof search exited because of Lean fatal error: {e}")
            search_result = SearchResult(
                theorem=self.theorem,
                status=Status.FAILED,
                proof=None,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            self._save_checkpoint(search_result)
            return search_result
        if self.root.status == Status.PROVED:
            proof = self.root.extract_proof()
        else:
            proof = None

        # extract a short version of the theorem for logging
        theorem_print = " ".join(
            list(filter(lambda x: "theorem" in x or "lemma" in x, self.theorem.split()))
        )
        result = SearchResult(
            theorem=self.theorem,
            status=Status.PROVED if self.is_valid_no_sorry else Status.FAILED,
            proof=proof,
            actor_time=self.actor_time,
            environment_time=self.environment_time,
            total_time=self.total_time,
            num_total_nodes=len(self.nodes),
            num_searched_nodes=self.num_expansions,
        )
        logger.info(f"Proving {[theorem_print]} finished, result: {result}")
        self._save_checkpoint(result)
        return result

    def sanity_check(self, proof: List[str]) -> bool:
        """
        Overview:
            Check the sanity of the proof.
        Arguments:
            - proof (List[str]): The proof to check.
        Returns:
            - bool: Whether the proof is sane.
        """
        code = self.theorem + proof
        codes = [{"code": code, "custom_id": str(uuid.uuid4())}]

        res = self.lean4_client.one_pass_verify_batch(codes, 60)

        try:
            res = res["results"][0]
            lean_feedback = res["response"]
            if not has_error(lean_feedback) and res["error"] is None:
                return True
        except Exception:
            return False
        return False

    def _search(self) -> None:
        """
        Overview:
            The internal function to search for proofs.
            Other search methods can extend this class and reimplement this function.
        """
        time_start = time.time()
        self.last_serialized = time_start

        self.priority_queue = queue.PriorityQueue()
        # for BestFirstSearch, the priority queue is a max-heap,
        # so we use negative priority (which is often the log-prob of the tactic)
        if self.priority_queue.empty():
            self.priority_queue.put((-self.root.priority, self.root))

        while True:
            if time.time() - self.last_serialized > self.serialize_interval:
                self._save_checkpoint()
                self.last_serialized = time.time()

            if self.priority_queue.empty():
                logger.info("Ran out of nodes to search.")
                time.sleep(0.1)
                break

            self.total_time = time.time() - time_start
            if self.total_time > self.timeout or (
                self.max_expansions is not None
                and self.num_expansions > self.max_expansions
            ):
                if self.root.status == Status.PROVED:
                    assert self.is_valid_no_sorry
                    logger.info(
                        "Found a proof! but hit the resource limit (timeout or max_expansions)."
                    )
                self.root.status = Status.OPEN
                logger.info("Hit the resource limit (timeout or max_expansions).")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                assert self.is_valid_no_sorry
                logger.info("Found a proof!")
                break

            self._expand_and_update()

    def _expand_and_update(self) -> None:
        """
        Overview:
            Perform a single step (expand and update) of search.
            Selects the node with the highest priority, queries the model for suggested tactics,
            and tries each tactic in the environment, creating and enqueuing a new node for each valid result.
            The expanded node is removed from the priority queue, and the new nodes are added to the priority queue.
        Arguments:
            - priority_queue (queue.PriorityQueue): The priority queue to store the nodes.
        """
        # Search the node with highest priority.
        try:
            _, search_node = self.priority_queue.get(timeout=1)
        except queue.Empty:
            return
        logger.debug(f"Expanding node: {search_node}")

        # Generate tactics for the current state, i.e. policy function.
        chosen_state = random.choice(search_node.codes)
        suggestions, code_prefix, indent = self._generate_tactics(chosen_state)
        # random generate tactics can be used for debugging
        # suggestions = await self._random_generate_tactics(search_node.state)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        updated_nodes = []
        tactic_result = self._run_seq_tactic(
            chosen_state, code_prefix, suggestions, indent
        )
        for states, is_valid_no_sorry, full_code in tactic_result:
            logger.debug(
                f"Verification Result [{is_valid_no_sorry}], full code: {[full_code]}"
            )
            logger.debug(f"Sequence tactic: {[full_code.split(code_prefix)[1]]}")
            nodes = self._construct_trees(states, search_node)
            if is_valid_no_sorry:
                self.is_valid_no_sorry = True
            updated_nodes.extend(nodes)
        # updated_nodes = list(set(updated_nodes))
        logger.debug(
            f"Total updated nodes: {len(updated_nodes)}, set updated nodes: {len(set(updated_nodes))}"
        )
        all_nodes = [node for node in self.nodes if isinstance(node, InternalNode)]
        logger.debug(
            f"Total internal nodes: {len(all_nodes)}, set internal nodes: {len(set(all_nodes))}"
        )

        # Keep only the leaf internal nodes.
        leaf_internal_nodes = list(
            filter(
                lambda node: all(
                    [not isinstance(edge.dst, InternalNode) for edge in node.out_edges]
                ),
                updated_nodes,
            )
        )
        logger.debug(f"Total leaf internal nodes: {len(leaf_internal_nodes)}")

        # Update the tree's status
        [node._recompute_status() for node in leaf_internal_nodes]
        [node._recompute_distance_to_proof() for node in leaf_internal_nodes]
        search_node.is_explored = True

        self.num_expansions += 1
        self.priority_queue.task_done()

        # If we're running in debug mode, run a full test suite each step
        # if self.debug:
        #     assert self.num_expansions == sum(
        #         node.is_explored
        #         for node in self.nodes.values()
        #         if isinstance(node, InternalNode)
        #     )
        #     self.check_invariants()

    def _construct_prompt(self, state: State) -> str:
        """
        Overview:
            Construct the prompt for the language model to generate tactics.
        Arguments:
            - state (State): The current lean state of the proof search.
        Returns:
            - prompt (str): The constructed prompt.
        """
        current_proof = reconstruct_proof(self.theorem, state)
        # prefix_text = "Complete the following Lean 4 code with " + \
        #     "explanatory comments preceding each line of code:\n\n```lean4\n"
        prefix_text = "Complete the following Lean 4 code:\n\n```lean4\n"
        prompt = prefix_text + current_proof["context"].rstrip()

        # indention = "".join([" "] * get_indention(prompt.split("\n")[-1]))
        indention = " " * 2
        first_indent = None
        for step in current_proof["steps"]:
            if "tactic" in step:
                prompt += step["tactic"]
                indention = "".join([" "] * get_indention(step["tactic"]))
                if first_indent is None:
                    first_indent = len(indention)
            else:
                prompt += f"\n{indention}/- tactic state:"
                for goal in step["goals"]:
                    format_goals = format_goals_with_indention(
                        goal, indention + " " * 2
                    )
                    prompt += f"\n{format_goals}"
                prompt += f"\n{indention}-/\n"
        code_prefix = prompt[len(prefix_text) :]
        if first_indent is None:
            first_indent = 2
        return prompt, code_prefix, first_indent

    def _generate_tactics(
        self, tactic_state: State
    ) -> Tuple[List[str, float], str, int]:
        """
        Overview:
            Generate tactics for the current state.
        Arguments:
            - tactic_state (State): The current lean state of the proof search.
        Returns:
            - List[Tuple[str, float]]: The generated tactics and their log-probabilities.
        """
        t0 = time.time()

        prompt, code_prefix, indent = self._construct_prompt(tactic_state)

        suggestions = self.tac_gen.generate(
            messages=self.tac_gen.build_messages(user_content=prompt),
            num_samples=self.num_sampled_tactics,
            prompt_type=self.prompt_type,
            temperature=self.temperature,
            max_length=self.max_length,
        )

        self.actor_time += time.time() - t0

        for sug in suggestions:
            logger.debug(f"Tactic suggestions: {[sug]}")
        logger.debug(f"Actor time: {self.actor_time:.2f}")
        return suggestions, code_prefix, indent

    def _run_seq_tactic(
        self,
        parent_state: State,
        code_prefix: str,
        suggestions: List[Tuple[str, float]],
        indent=2,
    ):
        """
        Apply a sequence of tactics to the current state and update the search tree.
        """
        if len(suggestions) == 0:
            return []
        elapsed_time = time.time()
        futures = []

        # debug
        # for seq_tactic, seq_logprob in suggestions:
        #     seq_tactic = seq_tactic.replace("```", "").rstrip()
        #     self.lean4_client.apply_seq_tactic(
        #         code_prefix, seq_tactic, seq_logprob, parent_state, 60, indent
        #     )

        with ThreadPoolExecutor(max_workers=len(suggestions)) as executor:
            for seq_tactic, seq_logprob in suggestions:
                seq_tactic = seq_tactic.replace("```", "").rstrip()
                futures.append(
                    executor.submit(
                        self.lean4_client.apply_seq_tactic,
                        code_prefix,
                        seq_tactic,
                        seq_logprob,
                        parent_state,
                        60,
                        indent,
                    )
                )

        results = []
        for future in as_completed(futures):
            states, is_valid_no_sorry, full_code = future.result()
            results.append((states, is_valid_no_sorry, full_code))

        elapsed_time = time.time() - elapsed_time
        logger.debug(f"[Environment time: {elapsed_time:2f}]")
        self.environment_time += elapsed_time

        return results

    def _construct_trees(self, states, node: InternalNode):
        """
        Construct the proof trees for the given states.
        """
        for state, logprob in states:
            logger.debug(f"Ran tactic {[state.tactics[-1]]} result in state: {state}")
            self.states.append(state)

        parent_node = node
        updated_nodes = []
        for state, logprob in states:
            is_old_node = False
            try:
                child_nodes = {
                    edge.dst.state: edge.dst for edge in parent_node.out_edges
                }
                result_node = child_nodes[state]
                is_old_node = True
            except KeyError:
                # Build a new node
                if state.is_valid:
                    if state.is_solved():
                        result_node = ProofFinishedNode(state)
                    else:
                        result_node = InternalNode(state)
                else:
                    result_node = ErrorNode(state)

                self.nodes.append(result_node)

            result_node.codes.append(state)
            if isinstance(result_node, InternalNode):
                result_node.cumulative_logprob = max(
                    result_node.cumulative_logprob,
                    logprob + parent_node.cumulative_logprob,
                )

            if (
                result_node.status == Status.OPEN and not is_old_node
            ):  # Don't search proved/failed nodes
                self.priority_queue.put((-result_node.priority, result_node))

            if isinstance(result_node, InternalNode) and not is_old_node:
                updated_nodes.append(result_node)

            # Build an edge connecting these nodes.
            # Will be added to the source node externally.
            if not is_old_node:
                edge = Edge(src=parent_node, dst=result_node)
                if isinstance(result_node, InternalNode):
                    result_node.in_edges.append(edge)

                assert isinstance(parent_node, InternalNode)
                parent_node.out_edges.append(edge)

            parent_node = result_node

        return updated_nodes

    def __del__(self) -> None:
        """
        Overview:
            This function is responsible for cleaning up resources when the object is garbage collected.
        """
        # self.lean4_client.close_context()
        pass

    #########
    # DEBUG #
    #########

    def check_invariants(self) -> None:
        """
        Overview:
            Perform some sanity checks.
        """
        for state, node in self.nodes.items():
            if state.is_solved():
                assert isinstance(node, ProofFinishedNode)
                assert self.root.status == Status.PROVED
            elif not state.is_valid:
                assert isinstance(node, ErrorNode)
            else:
                assert isinstance(node, InternalNode)
                node.check_invariants()

    def _random_generate_tactics(self, state: State) -> List[Tuple[str, float]]:
        """
        Overview:
            Randomly generate tactics for the current state, which is useful for debugging.
        Arguments:
            - state (State): The current lean state of the proof search.
        Returns:
            - List[Tuple[str, float]]: The generated tactics and their log-probabilities.
        """
        tactics = ["apply?", "rw?", "simp", "linarith", "norm_num", "ring", "aesop"]
        t0 = time.time()

        suggestions = random.choices(tactics, k=self.num_sampled_tactics)
        logprobs = [math.log(1 / len(tactics)) for _ in range(self.num_sampled_tactics)]
        suggestions = list(zip(suggestions, logprobs))

        self.actor_time += time.time() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions
