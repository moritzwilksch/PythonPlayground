# %%
import numpy as np
def calc_entropy(s):
    letter_counts = {letter: s.count(letter) for letter in s}
    l = len(s)
    probability_dict = {letter: letter_counts[letter]/l for letter in letter_counts}
    return {letter: - np.sum(probability_dict[letter] * np.log2(probability_dict[letter])) for letter in probability_dict}

# %%

print(calc_entropy("0000010100000"))
print(calc_entropy("1010111010110"))