from dotenv import load_dotenv
import os
import logging
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2

logging.basicConfig(level=logging.INFO)
load_dotenv()

llm_api_key = os.getenv("OPENROUTER_API_KEY")
model_name = "openai/gpt-4o-mini"
llm = dspy.MultiOpenAI(
    model=model_name,
    api_key=llm_api_key,
    api_provider="openrouter",
    model_type="chat",
    api_base="https://openrouter.ai/api/v1",
    max_tokens=300,
    # system_prompt=llama_3_system_prompt,
)
dspy.settings.configure(lm=llm)
# dspy.configure(lm=llm)

gms8k = GSM8K()

trainset, devset = gms8k.train, gms8k.dev


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


evaluate = Evaluate(
    devset=devset[:],
    metric=gsm8k_metric,
    num_threads=8,
    display_progress=True,
    display_table=False,
)

program = CoT()

evaluate(program, devset=devset[:])

# Initialize optimizer
teleprompter = MIPROv2(
    prompt_model=llm,
    task_model=llm,
    metric=gsm8k_metric,
    num_candidates=7,
    init_temperature=0.5,
    verbose=True,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    valset=devset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    minibatch=True,
    requires_permission_to_run=True,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized", save_field_meta=True)

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])
