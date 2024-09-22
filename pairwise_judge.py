from pydantic import BaseModel, Field
import dspy
import weave
from enum import StrEnum
from textwrap import dedent

system_prompt = """
You are an expert analyst of conversations. You are to analyze the a given question carefully and answer in `Yes` or `No`.
You should also provide a detailed explanation justifying your answer.
"""

llama_3_system_prompt = dedent(
    """
    Cutting Knowledge Date: December 2023
    Today Date: 21 September 2024

    When you receive a tool call response, use the output to format an answer to the orginal user question.

    You are a helpful assistant with tool calling capabilities.
    """
)

INPUT_KEYS = ["human_message", "conversation_a", "conversation_b"]


class ModelChoice(StrEnum):
    """The choice of model, either A or B"""

    A = "A"
    B = "B"


class ModelAnswer(StrEnum):
    """The winner of the pairwise evaluation"""

    tie = "tie"
    model_a = "model_a"
    model_b = "model_b"


class Input(BaseModel):
    human_message: str = Field(description="A message from a human to an AI")
    response_from_model_a: str = Field(description="The response from model A")
    response_from_model_b: str = Field(description="The response from model B")


class Output(BaseModel):
    choice: ModelChoice = Field(description="The choice of model")
    explanation: str = Field(description="The explanation for the choice of model")


class PairwiseSignature(dspy.Signature):
    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()


if __name__ == "__main__":
    import os
    import rich
    import configuration

    llm_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = "meta-llama/llama-3.1-70b-instruct"

    llm = dspy.MultiOpenAI(
        model=model_name,
        api_key=llm_api_key,
        api_provider="openrouter",
        model_type="chat",
        api_base="https://openrouter.ai/api/v1",
        max_tokens=512,
    )
    dspy.configure(lm=llm)

    dataset_ref = weave.ref("chatbot_arena_train").get()
    example_row = dataset_ref.rows[0]

    baseline_module = dspy.TypedPredictor(PairwiseSignature)
    example_input = Input(
        human_message=example_row["human_message"],
        response_from_model_a=example_row["response_a"],
        response_from_model_b=example_row["response_b"],
    )
    prediction = baseline_module(input=example_input)
    rich.print(prediction)
