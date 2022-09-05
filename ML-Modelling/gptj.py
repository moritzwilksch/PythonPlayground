#%%
from transformers import pipeline
from tqdm.auto import tqdm

pipe = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

#%%
pipe(
    "",
    max_new_tokens=100,
)
