import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap


X, color = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121, projection='3d')
sc1 = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Original Swiss Roll in 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.colorbar(sc1, ax=ax, shrink=0.5)


isomap = Isomap(n_components=2, n_neighbors=10)
X_reduced = isomap.fit_transform(X)

ax = fig.add_subplot(122)
sc2 = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll after IsoMap reduction")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
plt.colorbar(sc2, ax=ax, shrink=0.5)

plt.savefig('swiss_roll_isomap.png')

plt.show()
