import os
from dotenv import load_dotenv
import logging
import weave

logging.basicConfig(level=logging.INFO)

load_dotenv()

wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
weave.init(project_name=wandb_project_name)
