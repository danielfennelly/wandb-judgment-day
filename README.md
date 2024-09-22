# wandb-judgment-day
Let's hack on pairwise LLM evals

## Workflow

This project uses the toy problem of a joke bot model which will produce jokes for a given topic.

```
topic = "Office meetings"
jokebot_model = weave.ref(
    "weave:///danielfennelly/AB-Tests/object/JokeBot:LL6XBoNtdmUdJsYZ9gjDkTsLvqQx3AB4kExKNEX8bIE"
).get()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
print(write_joke(client, jokebot_model, topic))
```

> "Office meetings: where the only thing that gets accomplished is the reaffirmation that none of us are actually in charge. It's like a support group for people who can't make decisions on their own. 'Hi, my name is John, and I'm a middle manager.' 'Hi, John.' 'I've been in this meeting for 45 minutes and I still have no idea what we're discussing.' 'Welcome to the club, John. We have a PowerPoint presentation to prove it.'"

### 0. Setup

### 1. Create some models
```
python joke_models.py
```
### 2. Create some input data
```
python make_input_dataset.py
```

### 3. Create an output dataset to compare model outputs against each other
```
python make_output_datasets.py
```

### 4. Optimize an LLM-as-Judge from labelled training comparisons
*Under Construction*

### 5. Evaluate the Optimized Judge against the labelled test comparisons
*Under Construction*

### 6. Automate More Pairwise Tests
Profit!

## Label Studio

https://api.labelstud.io/tutorials/tutorials/evaluate-llm-responses#side-by-side-comparison

`docker compose --env-file .env up -d`