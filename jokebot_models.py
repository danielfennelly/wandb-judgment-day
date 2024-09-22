import weave
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

CLAUDE_SONNET_MODEL = "anthropic/claude-3.5-sonnet:beta"
LLAMA_3_70B_INSTRUCT = "meta-llama/llama-3.1-70b-instruct"
LLAMA_3_1_405B_INSTRUCT = "meta-llama/llama-3.1-405b-instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are a world famous comedian known for your intelligence and wit."
)
DEFAULT_INSTRUCTION_PROMPT = "Write a joke about {topic}."


@weave.op()
def write_joke(client, wmodel: weave.Model, topic: str) -> str:
    instruction = wmodel.instruction_prompt.format(topic=topic)
    response = client.chat.completions.create(
        model=wmodel.model_name,
        temperature=wmodel.temperature,
        messages=[
            {"role": "system", "content": wmodel.system_prompt},
            {"role": "user", "content": instruction},
        ],
    )
    return response.choices[0].message.content


class JokeBot(weave.Model):
    client: OpenAI = None
    model_name: str
    temperature: float
    system_prompt: str
    instruction_prompt: str

    @weave.op()
    def invoke(self, topic: str) -> str:
        return write_joke(self, topic)


if __name__ == "__main__":
    import os
    import configuration

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    jokebot = JokeBot(
        client=client,
        model_name=LLAMA_3_70B_INSTRUCT,
        temperature=0.4,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        instruction_prompt=DEFAULT_INSTRUCTION_PROMPT,
    )

    for model in [LLAMA_3_70B_INSTRUCT, LLAMA_3_1_405B_INSTRUCT, CLAUDE_SONNET_MODEL]:
        jokebot = JokeBot(
            client=client,
            model_name=model,
            temperature=0.4,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            instruction_prompt=DEFAULT_INSTRUCTION_PROMPT,
        )

        result = jokebot.invoke("dinosaurs")
        logger.info(result)
