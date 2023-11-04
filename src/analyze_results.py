import numpy as np
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", help="path to the results", default="src/results_data/")
parser.add_argument("--results_file", help="name of the results file", default="results_PPO_voltage_control_29032023.pkl")
args = parser.parse_args()

with open(os.path.join(args.results_path, args.results_file), "rb") as f:
    results_dict = pickle.load(f)

correlation = []
max_effect_consumer = []
max_obs = []
max_unchanged_state = []
action = []
for key, values in results_dict.items():
    correlation.append(values[0])
    max_effect_consumer.append(values[1])
    max_obs.append(values[2])
    max_unchanged_state.append(values[3])
    action.append(values[4])

print(action)

import matplotlib.pyplot as plt

timestamps = 96
# running average np.array(action).sum(axis=1)
plt.plot(np.convolve(np.array(action).sum(axis=1), np.ones((50,))/150, mode='valid')[:timestamps])
# plt.plot(np.array(action).sum(axis=1))
# plot max voltage
plt.plot(np.array(max_obs)[:timestamps])
plt.show()

# # do clusteing on the actions
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler


X = np.array(action)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
labels = kmeans.labels_
# plot the clusters in all the variables x
# do a pca on the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()




# do clustering on the correlation
X = np.array(correlation)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
labels = kmeans.labels_
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
