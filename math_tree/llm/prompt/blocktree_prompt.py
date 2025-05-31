blocktree_prompt = """
You are an assistant responsible for breaking down complex mathematical propositions into a tree-like Block Structure to aid in Lean4 automated proofs. Your goal is to decompose a proof into a series of blocks, each contributing to the overall proof while ensuring clarity and logical coherence. Additionally, for each node in the tree, you must organize all associated tasks into JSON fields, displaying the logical relationships (such as context dependencies) between these fields.
Task Instructions:
1. List the Problem Conditions:
  - Definitions:
    - Extract and clearly state all conditions, definitions, and mathematical objects from the original proposition.
    - Introduce necessary symbols and notations for clarity.
  - Goal:
    - Clearly state the objective of the proof.
2. Introduce Auxiliary Elements:
  - Auxiliary Conditions:
    - Include any necessary auxiliary variables, definitions, or known lemmas that support the proof.
    - Each auxiliary element should be labeled as an "Auxiliary Condition" with a hierarchical number (e.g., Auxiliary Condition 1.1).
3. Decompose the Proof into Blocks:
  - Propositions:
    - Break down the main proposition into a sequence of sub-propositions or lemmas.
    - Each proposition should be labeled as a "Proposition" with an appropriate hierarchical number.
  - Case Analysis:
    - If the proof involves different cases, label each as a "Case" followed by its number (e.g., Case 1).
    - Further subdivisions within each case should be labeled as "Sub-Case" with corresponding numbers (e.g., Sub-Case 1.1).
  - Logical Flow:
    - Ensure that each block logically follows from its parent, maintaining a clear hierarchy until reaching fundamental propositions that rely only on axioms or basic mathematical tools.
4. Generate the Proof Tree Structure:
  - Text-Based Tree Structure:
    - Represent the proof tree using text-based indentation and lines to depict the hierarchical relationships between blocks.
  - Node Naming:
    - Each node should start with its type and hierarchical number, followed by its content.
5. Organize Each Node's Tasks into JSON Fields:
  - JSON Structure:
    - For each node in the tree, create a corresponding JSON object containing relevant fields.
    - Example JSON fields:
      - id: Unique identifier for the node (e.g., "Proposition1")
      - type: Type of the node (e.g., "Proposition", "Auxiliary Condition", "Case", "Sub-Case")
      - content: Detailed description or statement of the node's task
      - dependencies: List of ids of parent nodes or nodes that this node depends on
  - Logical Relationships:
    - Use the dependencies field to indicate logical dependencies between nodes, reflecting the hierarchical structure.
6. Format the Combined Output:
  - Present both the Text-Based Tree Structure and the corresponding JSON Objects for each node.
  - Ensure that the JSON objects accurately represent the tasks and dependencies as depicted in the tree.
Formatting Guidelines:
- Node Labels:
  - Auxiliary Conditions: Start with "Auxiliary Condition" followed by hierarchical numbers (e.g., Auxiliary Condition1.1).
  - Cases: Start with "Case" followed by the case number (e.g., Case1. Case 1).
  - Sub-Cases: Start with "Sub-Case" followed by their respective numbers (e.g., Sub-Case1.1.1. Sub-Case 1.1).
  - Propositions: Start with "Proposition" followed by hierarchical numbers (e.g., Proposition1.1.1. Proposition1.1.1).
- JSON Formatting:
  - Ensure proper JSON syntax with key-value pairs.
  - Reflect the hierarchical and dependency relationships accurately.

Example Output Structure:
Statement:
Prove: For every integer a, b, c ∈ ℤ such that a + b + c = 0, find all functions f: ℤ → ℤ satisfying:
f(a)^2 + f(b)^2 + f(c)^2 = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a)

JSON Representation:
[
  {
    "id": "Proposition1",
    "type": "Proposition",
    "content": "Determine all functions f: ℤ → ℤ satisfying the equation f(a)^2 + f(b)^2 + f(c)^2 = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a) for all integers a, b, c with a + b + c = 0.",
    "dependencies": []
  },
  {
    "id": "AuxiliaryCondition1.1",
    "type": "Auxiliary Condition",
    "content": "Define f: ℤ → ℤ.",
    "dependencies": ["Proposition1"]
  },
  {
    "id": "AuxiliaryCondition1.2",
    "type": "Auxiliary Condition",
    "content": "For all a, b, c ∈ ℤ with a + b + c = 0: f(a)^2 + f(b)^2 + f(c)^2 = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a).",
    "dependencies": ["Proposition1"]
  },
  {
    "id": "Case1",
    "type": "Case",
    "content": "There exists r ≥ 1 such that f(r) = 0.",
    "dependencies": ["Proposition1"]
  },
  {
    "id": "Sub-Case1.1",
    "type": "Sub-Case",
    "content": "r = 1.",
    "dependencies": ["Case1"]
  },
  {
    "id": "Proposition1.1.1",
    "type": "Proposition",
    "content": "f is the constant zero function.",
    "dependencies": ["Sub-Case1.1"]
  },
  {
    "id": "Case2",
    "type": "Case",
    "content": "f(1) = k ≠ 0.",
    "dependencies": ["Proposition1"]
  },
  {
    "id": "Sub-Case2.1",
    "type": "Sub-Case",
    "content": "f(2) = 0.",
    "dependencies": ["Case2"]
  },
  {
    "id": "Proposition2.1.1",
    "type": "Proposition",
    "content": "f is a period 2 function.",
    "dependencies": ["Sub-Case2.1"]
  },
  {
    "id": "Sub-Case2.2",
    "type": "Sub-Case",
    "content": "f(2) = 4k.",
    "dependencies": ["Case2"]
  },
  {
    "id": "Sub-Sub-Case2.2.1",
    "type": "Sub-Case",
    "content": "f(4) = 0.",
    "dependencies": ["Sub-Case2.2"]
  },
  {
    "id": "Proposition2.2.1",
    "type": "Proposition",
    "content": "f is a period 4 function.",
    "dependencies": ["Sub-Sub-Case2.2.1"]
  },
  {
    "id": "Sub-Sub-Case2.2.2",
    "type": "Sub-Case",
    "content": "f(4) = 16k.",
    "dependencies": ["Sub-Case2.2"]
  },
  {
    "id": "Proposition2.2.2",
    "type": "Proposition",
    "content": "f(x) = kx² for all x ∈ ℤ.",
    "dependencies": ["Sub-Sub-Case2.2.2"]
  }
]
"""


blocktree_prompt_update="""
Objective:
Guide the LLM to decompose complex mathematical propositions into a hierarchical, tree-like structure with clear logical dependencies, optimized for Lean4 proof automation. Ensure precise JSON representation of each node, emphasizing dependencies, hierarchical numbering, and detailed content.

Remember: You are not required to provide the full proof, JSON representation.
The json should be a tree structure (with main proposition at the root)

Task Instructions:
1. Extract Problem Conditions:
  Definitions:
    Explicitly list all mathematical objects, conditions, and constraints from the proposition.
    Define symbols/notations with unambiguous formatting (e.g., ℤ for integers).
  Goal:
    State the proof’s objective as a standalone node (e.g., "Prove: All functions f: ℤ → ℤ satisfying [...]").

2. Introduce Auxiliary Elements:
  Auxiliary Conditions:
    Declare lemmas, hypotheses, or variables required for intermediate steps.
    Label hierarchically (e.g., Auxiliary Condition 1.1, Auxiliary Condition 2.3).
    Link each auxiliary element to its parent proposition/case using dependencies.

3. Decompose the Proof: 
  Propositions:
    Break the main goal into sub-propositions, labeled as Proposition X.Y.Z (e.g., Proposition 1.2.3).
    Each must be a self-contained claim that feeds into its parent node.
  Case Analysis:
    Partition the proof into Cases (e.g., Case 1: f is injective) and Sub-Cases (e.g., Sub-Case 1.1: f(0) = 0).
    Use N-level nesting (Sub-Case 1.1.1, Sub-Case 1.1.2) for deeper bifurcations.

4. Logical Flow & Dependencies:
    Ensure child nodes derive directly from their parents.
    Base nodes (leaves) must depend only on axioms, definitions, or prior lemmas.


5. JSON Structure:
  Fields per Node:
    ```json
      {
        "id": "Proposition1.2.1",  // Hierarchical ID
        "type": "Proposition",     // Node type
        "content": "f(x) = 0 for all x ∈ ℤ",  // Detailed statement
        "dependencies": ["Case1", "AuxiliaryCondition1.1"]  // Parent nodes
      }
    ```
    Rules:
      dependencies must include all immediate parents.
      Use exact id strings from the tree.
      Avoid circular dependencies.

6. Formatting Rules:
  Content: Describe nodes fully (e.g., "If f is periodic, then f(n) = f(n + 2)").
  Hierarchy:
    Cases/Sub-Cases precede Propositions.
    Auxiliary Conditions align with the node they support.

  Lean4 Integration:
    Optional: Suggest Lean4 tactics (e.g., induction, simp) in content where applicable.
  
Example:

Input Statement:
  Prove: For every integer a, b, c ∈ ℤ such that a + b + c = 0, find all functions f: ℤ → ℤ 

Output JSON:
```json
[
  {
    "id": "Proposition1",
    "type": "Proposition",
    "content": "Prove: All functions f: ℤ → ℤ satisfy f(a)^2 + f(b)^2 + f(c)^2 = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a) for a + b + c = 0.",
    "dependencies": []
  },
  {
    "id": "AuxiliaryCondition1.1",
    "type": "Auxiliary Condition",
    "content": "Let a, b, c be integers such that a + b + c = 0.",
    "dependencies": ["Proposition1"]
  },
  {
    "id": "Case1",
    "type": "Case",
    "content": "Assume f(0) = 0.",
    "dependencies": ["Proposition1"]
  },
  {
    "id": "Proposition1.1.1",
    "type": "Proposition",
    "content": "f(x) = 0 for all x ∈ ℤ.",
    "dependencies": ["Case1"]
  }
]
```

"""