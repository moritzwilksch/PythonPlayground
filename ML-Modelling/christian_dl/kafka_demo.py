# %%
from kafka import KafkaProducer
from kafka import KafkaConsumer
from rich import print
import re
import json

import nltk
import pandas as pd
import polars as pl
import ray
import tensorflow as tf
from nltk.stem import PorterStemmer
from ray.util.multiprocessing import Pool

pool = Pool()

ps = PorterStemmer()
stopwords = nltk.corpus.stopwords.words("german")

# %%
def prep_text(s: str) -> str:
    words = re.split(r"\s", s)
    words = [
        ps.stem(w)
        for w in words
        if w.lower() not in stopwords and len(w) > 2 and w != " "
    ]
    return " ".join(words)


def send(topic: str, message: str):
    producer = KafkaProducer(bootstrap_servers="localhost:9092")
    producer.send(topic, message.encode())


def consume(topic: str, group_id: str):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        group_id=group_id,
    )
    print("Listening...")
    for msg in consumer:
        print(msg)
        msg = json.loads(msg.value.decode())
        text = msg["descriptionText"]

        text = prep_text(text)
        s = tok.texts_to_sequences([text])
        print(s)
        s = tf.keras.utils.pad_sequences(s, maxlen=128)
        print(s)
        pred = model.predict(s)
        print(pred)
        send(
            "tender-prediction-response",
            json.dumps({"prediction": pred.tolist()[0][0], "tenderId": msg["tenderId"]}),
        )


consume("tender-prediction-request", "moritz0")
