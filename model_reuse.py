import os
from openai import OpenAI
import configuration
import weave
from weave.trace.refs import OpRef
from jokebot_models import write_joke

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
# from jokebot_models import *

loaded_model = weave.ref(
    "weave:///danielfennelly/AB-Tests/object/JokeBot:LL6XBoNtdmUdJsYZ9gjDkTsLvqQx3AB4kExKNEX8bIE"
).get()
joke = write_joke(client, loaded_model, "airplane food")

print(joke)
