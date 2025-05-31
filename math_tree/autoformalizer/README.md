# Pre commit hook

To ensure our codebase remains clean and consistent, we have integrated pre-commit hooks that automatically check and format the code according to defined standards. Before you can make commits, you need to set up the pre-commit hooks on your local machine. This is a one-time setup. Please run the following command in the root directory of this project:

```bash
sudo apt install pre-commit

pre-commit install
```

# Autoformalizer

Create the conda environment `autoformalization` with the following command to use this project:
```bash
conda env create -n autoformalization -f env.yml
conda activate autoformalization
``` 

Install the formalizer package:
```bash
pip install -e .
```

Log in to HF:
```bash
huggingface-cli login
```

# Setting up Lean

```bash
cd ~
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env
```

Choose the custom installation, type `y` to modify path variable, put `v4.11.0` for the toolchain version
then

```bash
git clone https://github.com/leanprover-community/repl.git && cd repl && git checkout adbbfcb9d4e61c12db96c45d227de92f21cc17dd
lake build
cd ..
```

then
```bash
git clone https://github.com/leanprover-community/mathlib4.git && cd mathlib4 && git checkout 20c73142afa995ac9c8fb80a9bb585a55ca38308
```

and finally
```bash
lake exe cache get && lake build
```

# Environment parameters

In order to make lean4 feedback work, we need to dump lean file locally.
c.f. https://github.com/project-numina/autoformalizer/blob/main/autoformalizer/eval_utils/constants.py

```bash
export AUTOFORMALIZER_WORKSPACE=your_workspace
```

your_workspace should be the directory that contains the `mathlib4` and `repl` installations.
