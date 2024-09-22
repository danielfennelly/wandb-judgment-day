import os
from dotenv import load_dotenv
import logging
import weave
import pickle
import dspy
from pairwise_judge import INPUT_KEYS, PairwiseSignature, Input, Output

logging.basicConfig(level=logging.INFO)

load_dotenv()

wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
weave.init(project_name=wandb_project_name)


def example_from_row(row):
    return {
        "input": Input(
            human_message=row["human_message"],
            response_from_model_a=row["response_a"],
            response_from_model_b=row["response_b"],
        ),
        "output": Output(choice=row["winner"][-1].upper(), explanation=""),
    }


train_dataset = weave.ref("chatbot_arena_train:v2").get()
dspy_train_examples = [
    dspy.Example(**example_from_row(row)).with_inputs("input")
    for row in train_dataset.rows
]
with open("dspy_train_examples.pkl", "wb") as f:
    pickle.dump(dspy_train_examples, f)
