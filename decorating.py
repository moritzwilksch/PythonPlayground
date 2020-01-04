# %% [markdown]
""" 
# Example of Decorating Python Functions
## Definition of Functions
"""
# %%
def func_output_to_upper(func):
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

def output_text():
    return "Lorem Ipsum blabla"

# %% [markdown]
# ## Decorating - the verbose way
# %%
# Normal way of decorating a function
test = func_output_to_upper(output_text)
print(test())

# %% [markdown]
# # Decorating - the easy way
# %%
# Decorator-way of decorating a function
@func_output_to_upper
def output_text_2():
    return "Second Lorem Ipsum blabla for real decoration" 

print(output_text_2())

# %% [markdown]
# ## Decorating *Functions* with Arguments
# %%
def print_call(func):
    def wrapper(arg1, arg2):
        print(f"{func.__name__}({arg1}, {arg2}) was called.")
        res = func(arg1, arg2)
        print("result =", res)
    return wrapper

@print_call
def multiply(a, b):
    return a*b

print(multiply(12,11))

# %% [markdown]
# ## *Decorators* with Arguments
# %%
def change_output_case(change_to_upper):
    def change_case(func):
        def wrapper():
            res = func()
            if change_to_upper:
                res = res.upper()
            else:
                res = res.lower()
            return res
        return wrapper
    return change_case

@change_output_case(change_to_upper=False)
def talk():
    return "This is a Text with MEssed-up Casing"
print(talk())

@change_output_case(change_to_upper=True)
def talk():
    return "This is a Text with MEssed-up Casing"
print(talk())

# %% [markdown]
# # Example: Timing Execution
# %%
import time

def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        print("Function", func.__name__, "(" + str(*args, **kwargs)+") returned:", res)
        print("--- %s seconds ---" % (time.time() - start_time))
    return wrapper

@time_execution
def multiply_until_n(n):
    prod = 1
    for i in range(1, n+1):
        prod = prod*i
    return prod

multiply_until_n(5)
multiply_until_n(500)