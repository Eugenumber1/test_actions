import pandas as pd
import pytest
from langevals import expect
from langevals_langevals.off_topic import (
    OffTopicSettings,
    OffTopicEvaluator,
    AllowedTopic,
)
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

entries = pd.DataFrame(
    {
        "input": [
            "What is unique about your product?",
            "Write a python code that prints fibonacci numbers.",
            "What are the core features of your platform?",
        ],
    }
)


@pytest.mark.parametrize("entry", entries.itertuples())
def test_extracts_the_right_address(entry):
    settings = OffTopicSettings(
        allowed_topics=[
            AllowedTopic(topic="our_product", description="Questions about our company and product"),
            AllowedTopic(topic="small_talk", description="Small talk questions"),
            AllowedTopic(topic="book_a_call", description="Requests to book a call"),
        ],
        model="openai/gpt-3.5-turbo-1106",
    )
    off_topic_evaluator = OffTopicEvaluator(settings=settings)
    expect(input=entry.input).to_pass(off_topic_evaluator)