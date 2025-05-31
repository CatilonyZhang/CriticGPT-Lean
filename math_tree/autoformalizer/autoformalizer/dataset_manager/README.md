We define here a list of datasets format to keep track of the data during the auto proving process.

### Base Dataset

These dataset store all the informal data. It is the starting point for the other dataset

List and the format of the dataset can be found in numinamath/datasets

Here are the important features of a base dataset for this pipeline:
- problem: the problem content in latex or markdown
- answer: final answer is the question_type is "math-word-problem". It is important for autoformalization. At the moment, answer is considered as reference, but it is not always the case in our dataset.
- problem_type: the mathematical domain of the problem. See find_problem_type for more information. Here are the supported types:
- question_type: The form or style of the mathematical problem.
    - MCQ
    - proof
    - math-word-problem: problem with output.
- id: ID on the app platform. It is used in the name of the theorem. Avoid using the id to index the data, because one might confuse this with an ordinary index.
- uuid: unique ID for every problem. You should make sure it is present on all the derivative datasets.


### Auto Problems Dataset
This is supposed to be the dataset to aggregate all formal data. TODO

For now, just consider this as a concatenation of the base datasets.

### Auto Statements Dataset

**Name convention** auto-statements-version
**Example** https://huggingface.co/datasets/AI-MO/auto-statements-v1

An auto statements dataset is used to store autoformalised statements. It should be the input of a auto proving pipeline.

Ideally, we apply filter out problems in the auto-problems dataset, for exemples, problems are not proved yet, to create a auto-statements dataset, and apply statement autoformalisation using https://github.com/project-numina/autoformalizer/blob/main/autoformalizer/model_utils/autoformalize_hf_dataset.py#L72

Here are the important features of a auto-statements dataset for this pipeline:

- uuid: the unique id
- problem: same as base dataset
- answer: same as base dataset
- theorem_names: list such as [problem_type_id]
- natural_language_statement: the exact natural language statement for autoformalisation. Should take into account the final answer

Once the prof search finish, metrics will be aggregate here as well
- n_proofs: number of proofs aggregated
- n_correct_proofs
- correct_proof_samples: a dump of a list of correct proofs 
- formal_proof: one of the correct proofs

### Auto Proofs dataset

Once we take the auto-statements dataset, apply any proof search (or sampling) pipeline, we get a auto-proofs dataset, where each proof is stored at a row.

**Example** https://huggingface.co/datasets/AI-MO/auto-proofs-v1

- uuid
- proof_id
- proof
- feedback: lean feedback from the server or lean repl
- is_valid: check if any error in the feedback
