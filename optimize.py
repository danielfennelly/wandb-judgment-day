import dspy
import weave
from dspy.evaluate import Evaluate
from dspy.teleprompt.signature_opt_typed import optimize_signature, OptimizerResult

from arena_dataset import Metadata
from pairwise_judge import INPUT_KEYS, PairwiseSignature, Input, Output
from pairwise_judge import llama_3_system_prompt


def example_from_row(row):
    return {
        "input": Input(
            human_message=row["human_message"],
            response_from_model_a=row["response_a"],
            response_from_model_b=row["response_b"],
        ),
        "output": Output(choice=row["winner"][-1].upper(), explanation=""),
    }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import logging

    # import pickle
    import configuration

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()

    llm_api_key = os.getenv("OPENROUTER_API_KEY")
    # model_name = "meta-llama/llama-3.1-70b-instruct"
    # model_name = "meta-llama/llama-3.1-405b-instruct"
    model_name = "openai/gpt-4o-mini"

    llm = dspy.MultiOpenAI(
        model=model_name,
        api_key=llm_api_key,
        api_provider="openrouter",
        model_type="chat",
        api_base="https://openrouter.ai/api/v1",
        max_tokens=8096,
        # system_prompt=llama_3_system_prompt,
    )
    dspy.configure(lm=llm)

    train_dataset = weave.ref("chatbot_arena_train:v2").get()
    dspy_train_examples = [
        dspy.Example(**example_from_row(row)).with_inputs("input")
        for row in train_dataset.rows
    ]

    baseline_module = dspy.TypedPredictor(PairwiseSignature)
    metadata = Metadata()

    @weave.op()
    def get_optimized_program(model: dspy.Module, metadata: Metadata) -> dspy.Module:

        @weave.op()
        def dspy_evaluation_metric(example, prediction, trace=None):
            return prediction.output.choice == example.output.choice

        evaluator = Evaluate(
            devset=dspy_train_examples,
            metric=dspy_evaluation_metric,
            num_threads=10,
            display_progress=True,
        )

        result: OptimizerResult = optimize_signature(
            student=baseline_module,
            evaluator=evaluator,
            initial_prompts=6,
            n_iterations=18,
            max_examples=metadata.max_labeled_demos,
            verbose=True,
        )
        return result.program

    optimized_module = get_optimized_program(baseline_module, metadata)
    optimized_module.save("optimized_module", save_field_meta=True)
