import pandas as pd
from loguru import logger

from autoformalizer.jobs.statement_formalizer import ShardProcessor
from autoformalizer.model_utils.infer_hf_dataset import negate_theorem


class Negator(ShardProcessor):

    def process_shard(
        self,
        shard_id,
        input_dir,
        output_dir,
        add_negation=True,
    ):
        """
        Negates theorems in the input shard and writes the result to the output shard.
        Assume input column name: formal_statement

        Input share require the following columns:
            - formal_statement: The formal statement of the theorem.
            - statement_: The statement of the theorem.
        """
        input_shard_path = input_dir / f"{shard_id}.parquet"
        output_shard_path = output_dir / f"{shard_id}.parquet"

        if not input_shard_path.exists():
            logger.error(f"Input shard {input_shard_path} does not exist.")
            return False, shard_id

        # Read the input shard
        df = pd.read_parquet(input_shard_path)

        if add_negation is False:
            # no negation required
            df.to_parquet(output_shard_path, index=False)
            logger.info(
                f"Shard {shard_id} saved to {output_shard_path} without negation"
            )
            return True, (shard_id, output_shard_path)

        negated_df = df.copy()
        negated_df["formal_statement"] = df["formal_statement"].apply(negate_theorem)
        # remove None or empty string
        negated_df = negated_df.dropna(subset=["formal_statement"])
        negated_df = negated_df[negated_df["formal_statement"] != ""]
        # make sure "theorem negated" is in statement
        negated_df = negated_df[
            negated_df["formal_statement"].str.contains("theorem negated")
        ]
        # log the number of entries removed
        logger.info(f"Shard {shard_id} removed {len(df) - len(negated_df)} entries")
        # add neg_ infront of the statement_id
        negated_df["statement_id"] = negated_df["statement_id"].apply(
            lambda x: f"neg_{x}"
        )

        # concatenate and reindex
        result_df = pd.concat([df, negated_df], ignore_index=True)

        result_df.to_parquet(output_shard_path, index=False)
        logger.info(f"Shard {shard_id} negated and saved to {output_shard_path}")
        return True, (shard_id, output_shard_path)
