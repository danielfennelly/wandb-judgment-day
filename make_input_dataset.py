import weave
from uuid import uuid4
import logging

import configuration

logger = logging.getLogger(__name__)

with open("joke_topics_train.csv", "r") as file:
    topics = file.readlines()


rows = [{"id": str(uuid4()), "topic": topic.strip()} for topic in topics]
for row in rows:
    logger.info(row)


weave.publish(weave.Dataset(name="jokebot_input_train", rows=rows))

with open("joke_topics_test.csv", "r") as file:
    topics = file.readlines()


rows = [{"id": str(uuid4()), "topic": topic.strip()} for topic in topics]
for row in rows:
    logger.info(row)


weave.publish(weave.Dataset(name="jokebot_input_test", rows=rows))
