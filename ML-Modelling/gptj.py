#%%
from transformers import pipeline
from tqdm.auto import tqdm

pipe = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

#%%
pipe(
    """ 
    # return the greater of the two numbers.
    def greater_number(a, b):
        return
    """,
    max_new_tokens=50,
)
