import os
import dspy
import weave
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
weave.init(project_name=wandb_project_name)


llm_api_key = os.getenv("OPENROUTER_API_KEY")
model_name = "meta-llama/llama-3.1-70b-instruct"

openrouter_lm = dspy.MultiOpenAI(
    model=model_name,
    api_key=llm_api_key,
    api_provider="openrouter",
    model_type="chat",
    api_base="https://openrouter.ai/api/v1",
    max_tokens=512,
)
dspy.configure(lm=openrouter_lm)


class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context = dspy.InputField(desc="facts here are assumed to be true")
    text = dspy.InputField()
    faithfulness = dspy.OutputField(
        desc="True/False indicating if text is faithful to context"
    )


class WeaveModel(weave.Model):
    signature: type

    @weave.op()
    def predict(self, context: str, text: str) -> bool:
        return dspy.ChainOfThought(self.signature)(context=context, text=text)


context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
text = "Lee scored 3 goals for Colchester United."

model = WeaveModel(name="Weave DSPy Example", signature=CheckCitationFaithfulness)
print(model.predict(context, text))
