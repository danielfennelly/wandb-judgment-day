import os

import weave
from datasets import load_dataset


class Metadata(weave.Object):
    dataset_address: str = "lmsys/chatbot_arena_conversations"
    num_train_examples: int = 50
    num_val_examples: int = 50
    split: str = "train"  # There's only one "split" for this dataset
    shuffle_seed: int = 4
    model: str = "meta-llama/llama-3.1-70b-instruct"
    max_tokens: int = 4096
    max_bootstrapped_demos: int = 8
    max_labeled_demos: int = 8


def reshape_row(row) -> dict:
    id = row["question_id"]
    human_message = row["conversation_a"][0]["content"]
    response_a = row["conversation_a"][1]["content"]
    response_b = row["conversation_b"][1]["content"]
    winner = row["winner"]
    return {
        "id": id,
        "winner": winner,
        "human_message": human_message,
        "response_a": response_a,
        "response_b": response_b,
    }


unambiguous_winners = ["model_a", "model_b"]


@weave.op()
def etl_dataset(metadata: Metadata, hf_token: str = None):
    dataset = load_dataset(
        metadata.dataset_address, split=metadata.split, streaming=True, token=hf_token
    )
    dataset = dataset.shuffle(seed=metadata.shuffle_seed).filter(
        lambda row: row["turn"] == 1 and (row["winner"] in unambiguous_winners)
    )

    # create the training and validation datasets
    train_rows = list(dataset.take(metadata.num_train_examples))
    train_rows = list(map(reshape_row, train_rows))
    val_rows = list(dataset.take(metadata.num_val_examples))
    val_rows = list(map(reshape_row, val_rows))
    # publish the datasets to the Weave, this would let us version the data and use for evaluation
    weave.publish(weave.Dataset(name="chatbot_arena_train", rows=train_rows))
    weave.publish(weave.Dataset(name="chatbot_arena_validate", rows=val_rows))


if __name__ == "__main__":
    import configuration

    # Enable Tracking

    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
    weave.init(project_name=wandb_project_name)

    hf_token = os.getenv("HF_TOKEN")
    metadata = Metadata()
    etl_dataset(metadata, hf_token=hf_token)
