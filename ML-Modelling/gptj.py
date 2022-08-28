#%%
from transformers import pipeline
from tqdm.auto import tqdm

pipe = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

#%%
pipe(
    """
    Generate domain names for a technology startup.
    domain: www.zwei.dev
    domain: www.testr.io
    domain: www.
    """,
    max_new_tokens=100,
)

#%%
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

df = fetch_20newsgroups(
    subset="train", categories=["sci.med"], shuffle=True, random_state=42
)

df = pd.DataFrame(df)



