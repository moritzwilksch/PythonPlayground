#%%
import polars as pl

#%%
df = pl.DataFrame(
    {
        "a": [False, False, True, True],
        "b": [False, True, False, True],
    }
)

df.select(pl.any(["a", "b"]))

#%%
df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6], "bar": ["a", "b", "c", "d", "e", "f"]})
df.to_arrow()

#%%
df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
df.quantile(0.7).collect()


#%%
df = pl.DataFrame(
    {"letters": ["a", "a", "b", "c"], "numbers": [[1], [2, 3], [4, 5], [6, 7, 8]]}
).lazy()
result = df.explode("numbers").filter(pl.col("numbers") < 6)
print(result.describe_optimized_plan())
