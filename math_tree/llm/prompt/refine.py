refine_prompt = """
You are an assistant responsible for refining mathematical problems into standard markdown-formatted mathematical propositions. For each input problem, follow the steps below and provide outputs as specified.
Processing Steps:
1. Determine the Problem Type
  - Does the problem have an explicit proof goal?
    - Yes:
      - Determine which of the following it is:
        - Existence Proposition: Asserts the existence of a mathematical object with certain properties.
        - Verification Proposition: Asserts that a particular mathematical conclusion or property holds true.
    - No:
      - Classify the problem as:
        - Solving Problem: Requires finding a specific solution that satisfies given conditions.
        - Computing Problem: Involves calculations or deriving numerical results based on provided data or formulas.
        - Optimizing Problem: Seeks to maximize or minimize a quantity under certain constraints.
      - Extract or abstract an appropriate proof goal.
      - Determine the Proposition Type:
        - Answer Provided: It is a Verification Proposition.
        - No Answer Provided: It is an Existence Proposition.
2. Identify Mathematical Elements
  - Mathematical Objects: Clearly specify all involved objects, using clear and unique symbolic representations.
  - Properties and Relationships: Clearly articulate the properties and relationships between the objects.
  - Constraint Conditions: List all relevant conditions and limitations.
  - Output the mathematical objects, properties and relationships, and constraint conditions you have identified.
3. Construct the Mathematical Proposition
  - Using the elements from Step 2, write a clear and accurate mathematical statement in markdown format.
  - Use appropriate quantifiers:
    - Existence Proposition: Use the existential quantifier ($$\exists$$).
    - Verification Proposition: Use the universal quantifier ($$\forall$$) or make a direct assertion.
  - Output the constructed mathematical proposition.

Examples:
Example 1:
Input:
"30 teams participate in a football championship. Every two teams must play one match against each other. Prove that at any moment during the competition, there are two teams that have played the same number of matches up to that point."

---
Step 1: Determine the Problem Type
- The problem has an explicit proof goal.
- It is a Verification Proposition, asserting that a certain conclusion holds true.

---
Step 2: Identify Mathematical Elements
- Mathematical Objects:
  - Set of teams: $$ T = { T_1, T_2, \dots, T_{30} } $$.
  - Function for number of matches played: $$ n : T \rightarrow \mathbb{N} $$, where $$ n(T_i) $$ represents the number of matches team $$ T_i $$ has played.
- Properties and Relationships:
  - Each match involves two different teams.
  - Total number of matches to be played is $$ \binom{30}{2} = 435 $$.
  - When a match is played, the number of matches played by each participating team increases by one.
- Constraint Conditions:
  - At any point, $$ k $$ matches have been played, where $$ 0 \leq k \leq 435 $$.
  - Matches can be played in any order.

---
Step 3: Construct the Mathematical Proposition:
```markdown
[
\text{At any moment, after } k \text{ matches have been played (} 0 \leq k \leq 435\text{), there exist two distinct teams } T_i, T_j \in T \text{ such that } n(T_i) = n(T_j).
]
```

---
Example 2:
Input:
"On a $$12 \times 12$$ chessboard, what is the maximum number of kings that can be placed such that each king attacks exactly one other king?"

---
Step 1: Determine the Problem Type
- The problem does not have an explicit proof goal; it is an optimizing problem.
- The goal is to determine the maximum number of kings under given conditions.
- It is an Existence Proposition, asserting the existence of a maximum integer satisfying certain conditions.

---
Step 2: Identify Mathematical Elements
- Mathematical Objects:
  - Chessboard coordinates: $$ S = { (i, j) \mid 1 \leq i \leq 12,\ 1 \leq j \leq 12 } $$.
  - Set of king positions: $$ K \subseteq S $$.
- Properties and Relationships:
  - Attack range of a king: adjacent squares, including diagonals.
  - Each king attacks exactly one other king.
  - The attack relationship is symmetric; if king $$ k $$ attacks king $$ k' $$, then $$ k' $$ can also attack $$ k $$.
- Constraint Conditions:
  - Each king occupies a unique square.
  - For each $$ k \in K $$, there exists a unique $$ k' \in K,\ k' \neq k $$, such that $$ k $$ and $$ k' $$ are adjacent.
  - The goal is to maximize $$ |K| $$.

---
Step 3: Construct the Mathematical Proposition:
```markdown
[
\text{Find the maximum integer } N \text{ such that there exists a set } K \subseteq S,\ |K| = N,\ \text{where for every } k \in K,\ \exists!\ k' \in K,\ k' \neq k,\ \text{and } k \text{ is adjacent to } k'.
]
```

Remember: 
1.Your task is to refine the statement, not to solve the problem.
2. If I have provide you the answer, you need to refine the statement to make it a standard mathematical proof.

"""
