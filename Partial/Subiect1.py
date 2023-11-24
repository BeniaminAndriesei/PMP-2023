import random
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


def aruncare_moneda(probabilitate_stema):
    return random.random() < probabilitate_stema

def joc():
    # Decide cine incepe
    incepe_p0 = random.choice([True, False])

    # Moneda lui P0
    if aruncare_moneda(1/3):
        steme_p0 = 1
    else:
        steme_p0 = 0

    # Moneda lui P1
    steme_p1 = sum([1 for _ in range(steme_p0 + 1) if aruncare_moneda(0.5)])

    # Determina castigatorul
    castigator = 0 if steme_p0 >= steme_p1 else 1

    return castigator if incepe_p0 else 1 - castigator

# Simuleaza jocul de 20,000 de ori
numar_simulari = 20000
castiguri_p0 = sum([joc() for _ in range(numar_simulari)])

# Calculeaza procentajele de castig
procentaj_p0 = castiguri_p0 / numar_simulari * 100
procentaj_p1 = 100 - procentaj_p0

print(f"Jucatorul P0 are o sansa de castig de {procentaj_p0:.2f}%")
print(f"Jucatorul P1 are o sansa de castig de {procentaj_p1:.2f}%")





# Definirea rețelei bayesiene
model = BayesianModel([('P0_moneda', 'P0_steme'), ('P1_moneda', 'P1_steme'), ('P0_steme', 'castigator')])

# Estimarea parametrilor pe baza datelor simulare
date_simulare = []
for _ in range(numar_simulari):
    incepe_p0 = random.choice([True, False])
    p0_moneda = aruncare_moneda(1/3)
    p1_moneda = aruncare_moneda(0.5)
    p0_steme = 1 if p0_moneda else 0
    p1_steme = sum([1 for _ in range(p0_steme + 1) if aruncare_moneda(0.5)])
    castigator = 0 if p0_steme >= p1_steme else 1
    date_simulare.append({'P0_moneda': p0_moneda, 'P0_steme': p0_steme, 'P1_moneda': p1_moneda, 'P1_steme': p1_steme, 'castigator': castigator})

# Ajustarea modelului la datele simulate
model.fit(date_simulare, estimator=MaximumLikelihoodEstimator)

# Determinarea feței monedei mai probabile în prima rundă, știind că în a doua rundă nu s-au obținut steme
inference = VariableElimination(model)
rezultate_inferenta = inference.map_query(variables=['P0_moneda'], evidence={'P1_steme': 0})
probabilitate_fata = rezultate_inferenta.values[0]

print(f"Fata monedei lui P0 este mai probabil să fie: {'Stema' if probabilitate_fata > 0.5 else 'Cap'} cu o probabilitate de {probabilitate_fata:.2f}")
