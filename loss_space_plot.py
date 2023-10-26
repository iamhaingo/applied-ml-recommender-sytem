import matplotlib.pyplot as plt
import numpy as np


tau = 0.01
lambd = 0.001
r = 5

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)


Pu = np.exp(-0.5 * tau * X**2)
Pv = np.exp(-0.5 * tau * Y**2)
Puv = np.exp(-0.5 * lambd * (r - X * Y) ** 2)

Z = Pu * Pv * Puv

plt.figure(figsize=(16, 8))
plt.xlabel("X")
plt.ylabel("Y")

plt.contourf(X, Y, Z)
plt.axis("square")
plt.show()
