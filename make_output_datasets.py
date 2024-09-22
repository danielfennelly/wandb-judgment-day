import os
import weave
from weave import Dataset
import logging
from label_studio_sdk.client import LabelStudio
from jokebot_models import write_joke

logger = logging.getLogger(__name__)

with open("label_studio_comparison_view.xml", "r") as view_file:
    comparison_view: str = view_file.read()


def find_project(ls: LabelStudio, project_name: str):
    for project in ls.projects.list():
        if project.title == project_name:
            return project
    return None


def get_or_create_project(ls: LabelStudio, project_name: str):
    project = find_project(ls, project_name)
    if project is None:
        project = ls.projects.create(
            title=project_name,
            description="Pairwise comparison of LLM responses",
            label_config=comparison_view,
        )
    return project


if __name__ == "__main__":
    import os
    from openai import OpenAI

    import configuration

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    logger.info("Creating comparison datasets")
    model1_name = "JokeBot:v2"  # Note: not sure why I can't use this as the ref
    model1 = weave.ref(
        "weave:///danielfennelly/AB-Tests/object/JokeBot:LL6XBoNtdmUdJsYZ9gjDkTsLvqQx3AB4kExKNEX8bIE"
    ).get()
    model2_name = "JokeBot:v3"
    model2 = weave.ref(
        "weave:///danielfennelly/AB-Tests/object/JokeBot:TlXvulXSWwfH7rpD4GCsjlyfO4UMcc5DRfxp2Gexngk"
    ).get()

    ls = LabelStudio(
        base_url=os.getenv("LABEL_STUDIO_URL"),
        api_key=os.getenv("LABEL_STUDIO_API_KEY"),
    )
    ls_project_name = "Weights and Biases: Judgment Day"
    project = get_or_create_project(ls, ls_project_name)
    labelling_fstring = "Model was prompted to write a joke about: {}"

    # TODO: This is embarrassingly parallelizable but it's a hackathon so w/e
    def generate_labelling_data(dataset_name: str):
        dataset = weave.ref(dataset_name).get()
        labelling_tasks = []
        for row in dataset.rows:
            (id, topic) = (row["id"], row["topic"])
            joke1 = write_joke(client, model1, topic)
            joke2 = write_joke(client, model2, topic)
            task = {
                "id": id,
                "topic": topic,
                "prompt": labelling_fstring.format(topic),
                "answer1": joke1,
                "answer1_model": model1_name,
                "answer2": joke2,
                "answer2_model": model2_name,
                "source": dataset_name,
            }
            labelling_tasks.append(task)
        return labelling_tasks

    # Make Comparison Outputs (Train)
    logger.info("Generating labelling data for training")
    labelling_data_train = generate_labelling_data("jokebot_input_train")
    comparison_dataset_train = Dataset(
        name="jokebot_comparison_train",
        rows=labelling_data_train,
    )
    weave.publish(comparison_dataset_train)
    ls.projects.import_tasks(id=project.id, request=labelling_data_train)

    # Make Comparison Outputs (Test)
    logger.info("Generating labelling data for test")
    labelling_data_test = generate_labelling_data("jokebot_input_test")
    comparison_dataset_test = Dataset(
        name="jokebot_comparison_test",
        rows=labelling_data_test,
    )
    weave.publish(comparison_dataset_test)
    ls.projects.import_tasks(id=project.id, request=labelling_data_test)
