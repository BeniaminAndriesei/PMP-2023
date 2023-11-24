import pymc3 as pm
import numpy as np

# Generarea setului de date
data = np.random.normal(loc=u_true, scale=q_true, size=200)

# Definirea modelului PyMC
with pm.Model() as wait_time_model:
    u = pm.Normal('u', mu=u_prior_mean, sd=u_prior_std)
    q = pm.Normal('q', mu=q_prior_mean, sd=q_prior_std)
    observed_data = pm.Normal('observed_data', mu=u, sd=q, observed=data)

    #  Alegerea distribuțiilor a priori depinde de cunoștințele anterioare pe care le avem despre timpul
    #  de așteptare. Dacă nu avem cunoștințe anterioare, putem alege distribuții cu deviații standard 
    #  relativ largi pentru a exprima incertitudinea.
    

    # Alegerea distribuțiilor a priori
u_prior_mean = 0  # Valoarea medie a priori pentru u
u_prior_std = 10  # Deviația standard a priori pentru u

q_prior_mean = 5  # Valoarea medie a priori pentru q
q_prior_std = 2   # Deviația standard a priori pentru q


# distribuția a posteriori pentru parametrul q.
with wait_time_model:
    trace = pm.sample(2000, tune=1000, cores=2) 

pm.plot_posterior(trace, var_names=['q'])
