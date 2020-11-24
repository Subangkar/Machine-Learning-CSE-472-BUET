import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from models.PCA import pca_2

# %%
from models.EMGauss import EMGauss


# %%
def plot_2d(X):
    # plt.title('Standardized Data')
    plt.figure(figsize=(7, 7))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.xticks()
    # plt.yticks()
    plt.scatter(X[:, 0], X[:, 1], color='blue', marker='o')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True, which='both')

    # plt.axhline(y=0, color='k')
    # plt.axvline(x=0, color='k')
    plt.show()

    # sns.scatterplot(X_t[:, 0], X_t[:, 1], markers='o')
    # plt.title('Scatter-plot')
    # plt.show()


# %%
X = np.genfromtxt('data/data.txt', delimiter=' ')
print(X.shape)

X_t = pca_2(X)
print(X_t.shape)

plt.figure(figsize=(7, 7))
sns.scatterplot(X_t[:, 0], X_t[:, 1])
plt.title('PCA-plot')
plt.show()

# %%
rgb2hex = lambda r, g, b: f"#{r:02x}{g:02x}{b:02x}"

model = EMGauss(n_components=3, n_iterations=100, init='kmeans', early_stop=True)
model.fit(X_t, verbose=True, verbose_freq=10)
labels = model.predict(X_t)  # np.max(model.predict_proba(X), axis=1)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X_t[:, 0], X_t[:, 1],
                     c=labels, s=50)
ax.scatter(model.mu[:, 0], model.mu[:, 1], c='red', s=700, marker='*')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
# plt.colorbar(scatter)

model.print_parameters()
