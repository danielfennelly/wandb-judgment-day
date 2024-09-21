import argparse
import os
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
import weave
from weave import Dataset as WeaveDataset
import json
from typing import Optional, List
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(
    hf_dataset: str,
    hf_token: str,
    wandb_project: str,
    wandb_dataset: Optional[str] = None,
    count: int = 10,
    preview: bool = False,
    splits: List[str] = ["train", "test"],
    seed: int = 4,  # Note: chosen randomly ;)
):

    dataset: Dataset = load_dataset(
        hf_dataset,
        split="train",
        streaming=True,
        token=hf_token,
    )

    # Constrain to only 1-turn conversations for data consistency
    dataset = dataset.shuffle(seed=seed).filter(lambda row: row["turn"] == 1)

    if splits:
        rows = {}
        for split in splits:
            rows[split] = list(dataset.take(count))
    else:
        rows = list(dataset.take(count))

    if preview:
        logging.info(json.dumps(dataset.info._to_yaml_dict(), indent=2))
        logging.info(
            json.dumps(rows[0] if not splits else rows[splits[0]][0], indent=2)
        )
        return

    weave.init(project_name=wandb_project)

    if not wandb_dataset:
        wandb_dataset = f"{hf_dataset}-{uuid4()}"

    # Create a dataset
    if splits:
        for split in splits:
            split_rows = rows[split]
            name = f"{wandb_dataset}-{split}"
            logger.info("Creating dataset %s" % name)
            dataset = WeaveDataset(
                name=name, description=f"Loaded from ${hf_dataset}", rows=split_rows
            )
            # Publish the dataset
            logger.info("Publishing dataset %s" % name)
            weave.publish(dataset)

    else:
        logger.info("Creating dataset %s" % wandb_dataset)
        dataset = WeaveDataset(
            name=wandb_dataset,
            description=f"Loaded from ${hf_dataset}",
            rows=rows,
        )
        # Publish the dataset
        logger.info("Publishing dataset %s" % wandb_dataset)
        weave.publish(dataset)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="HF to W&B ETL",
    )
    # preview flag to preview the dataset
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the dataset",
    )
    # dataset name
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset name",
        default="lmsys/chatbot_arena_conversations",
    )
    parser.add_argument(
        "--wandb-project", "-p", type=str, help="W&B project name", default="AB-Tests"
    )
    parser.add_argument(
        "--wandb-dataset", "-w", type=str, help="W&B dataset name", default="example"
    )
    parser.add_argument(
        "--count", "-c", type=int, help="Number of samples to load", default=10
    )

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN environment variable")
    wandb_token = os.getenv("WANDB_API_KEY")
    if not wandb_token:
        raise ValueError("Please set the WANDB_API_KEY environment variable")

    args = parser.parse_args()
    main(
        hf_dataset=args.dataset,
        wandb_project=args.wandb_project,
        hf_token=hf_token,
        count=args.count,
        preview=args.preview,
    )
