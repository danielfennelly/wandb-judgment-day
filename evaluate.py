import weave
import dspy
import asyncio

from pairwise_judge import Input, Output, PairwiseSignature


@weave.op()
def weave_evaluation_scorer(winner: str, model_output: dict) -> dict:
    # note: fix data loading to use ModelChoice enum to avoid this jank
    return {"match": int(winner[-1].lower() == model_output["choice"].lower())}


if __name__ == "__main__":
    import os
    import configuration

    llm_api_key = os.getenv("OPENROUTER_API_KEY")
    # model_name = "meta-llama/llama-3.1-70b-instruct"
    model_name = "openai/gpt-4o-mini"

    llm = dspy.MultiOpenAI(
        model=model_name,
        api_key=llm_api_key,
        api_provider="openrouter",
        model_type="chat",
        api_base="https://openrouter.ai/api/v1",
        max_tokens=512,
    )
    dspy.configure(lm=llm)

    validation_dataset = weave.ref("chatbot_arena_validate:v2").get()

    baseline_module = dspy.TypedPredictor(PairwiseSignature)
    baseline_module.load("optimized_module")

    @weave.op()
    def predict(human_message, response_a, response_b) -> dict:
        input = Input(
            human_message=human_message,
            response_from_model_a=response_a,
            response_from_model_b=response_b,
        )
        return baseline_module(input=input).output.dict()

    evaluation = weave.Evaluation(
        name="baseline_chat_arena_module",
        dataset=validation_dataset,
        scorers=[weave_evaluation_scorer],
    )
    asyncio.run(evaluation.evaluate(predict))
