import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the Seaborn style
sns.set(style="whitegrid", font_scale=2)
sns.set_style("ticks")
sns.color_palette("Set1")
# sns.set_context("talk")


lambd = 0.01
r = 5
tau_values = [0.01, 0.5]  # List of tau values

# Make data.
X = np.arange(-5, 5, 0.01)
Y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(X, Y)

plt.figure(figsize=(16, 8))

for tau in tau_values:
    Pu = np.exp(-0.5 * tau * X**2)
    Pv = np.exp(-0.5 * tau * Y**2)
    Puv = np.exp(-0.5 * lambd * (r - X * Y) ** 2)

    Z = Pu * Pv * Puv

    plt.subplot(1, len(tau_values), tau_values.index(tau) + 1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(rf"$\tau$ = {tau}")

    plt.contourf(X, Y, Z)
    plt.axis("square")

    # Add lambda value with LaTeX symbol
    plt.text(3, 4, rf"$\lambda = {lambd}$", color="white", fontsize=12)

plt.tight_layout()
plt.savefig("./tau-lambda.pdf")
plt.show()
