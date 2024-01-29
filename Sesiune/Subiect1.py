import pandas as pd
import pymc3 as pm
import numpy as np

# incarc setul de date
df = pd.read_csv('Boston Housing.csv', decimal=',')

# modelul în PyMC
with pm.Model() as model:
    # variabilele independente
    rm = pm.Data('rm', df['rm'])
    crim = pm.Data('crim', df['crim'])
    indus = pm.Data('indus', df['indus'])

    # variabila dependentă
    medv = pm.Data('medv', df['medv'])

    # parametrii pentru intercept și coeficienti
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta_rm = pm.Normal('beta_rm', mu=0, sd=10)
    beta_crim = pm.Normal('beta_crim', mu=0, sd=10)
    beta_indus = pm.Normal('beta_indus', mu=0, sd=10)

    # modelul liniar
    mu = alpha + beta_rm * rm + beta_crim * crim + beta_indus * indus

    #distributia pentru variabila dependenta
    medv_obs = pm.Normal('medv_obs', mu=mu, sd=1, observed=medv)

# Estimez parametrii modelului pentru a obtine HDI
with model:
    trace = pm.sample(1000, tune=1000)

# estimari de 95% HDI pentru parametrii
summary = pm.summary(trace, hdi_prob=0.95)
print(summary)

# Simulez extrageri din distributia predictiva posterioara
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=1000)

# Calculez intervalul de predicție de 50% HDI pentru valoarea locuintelor
y_pred = post_pred['medv_obs']
hdi_50 = pm.stats.hpd(y_pred.flatten(), hdi_prob=0.5)

# Afisez intervalul de predicție de 50% HDI
print("Interval de predicție de 50% HDI pentru valoarea locuintelor:", hdi_50)

