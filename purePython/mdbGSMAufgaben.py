# %%
import numpy as np
# %%
r900 = [10, 20]
r1800 = [2, 5]
#[2.5*r**2 for r in r900 + r1800]
# %%
namen = ["Potsdam", "China", "Campus Griebnitzsee", "Digitalvilla"]
a_zelle = [2.5*r**2 for r in r900 + r1800]
a_ges = [a * n*10 for a in a_zelle for n in range(5, 20, 5)]
user = [int(320*n*a/10) for n in range(10, 200, 10) for a in a_zelle]
percent = [0.1, 0.2, 0.25]
population = [u/p for u in user for p in percent]
bts_cost = [1000, 2000, 4000, 5000, 10000]
# %%
def generate_aufgabe():
    s = f"""
    Im Fürstentum {np.random.choice(namen)} leben {np.random.choice(population)} Menschen.
    Diese verteilen sich gleichmäßig auf {np.random.choice(a_ges)} km^2.
    {np.random.choice(percent)*100}% der Personen wollen das Netz gleichzeitig Nutzen.

    Ihnen liegen folgende Angebote vor:

    GSM900 Einheit mit 56 Kanälen und maximaler Reichweite von {np.random.choice(r900)} km für {np.random.choice(bts_cost)}€
    GSM1800 Einheit mit 140 Kanälen und maximaler Reichweite von {np.random.choice(r1800)} km für {np.random.choice(bts_cost)}€
    =============================================================
    """
    return s

for _ in range(10):
    print(generate_aufgabe())