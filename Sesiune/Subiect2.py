
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem with a geometric distribution
    for the first appearance of heads.
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    # Use geom.pmf with all possible values of k
    likelihood = stats.geom.pmf(np.arange(1, grid_points + 1), grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

# Caut valorea lui "theta" care maximizeaza probabilitatea a posteriori
argmax_theta = grid[np.argmax(posterior)]
print(f"Theta that maximizes the posterior probability: {argmax_theta}")
 