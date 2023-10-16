import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

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

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Plot")
plt.show()
