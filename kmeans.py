from collections import defaultdict
from functools import reduce
import numpy as np
import random
import matplotlib.pyplot as plt
from pprint import pprint
import plotly.express as px

def euclidean(x, y):
    return np.sum(np.sqrt([(x1 - x2) ** 2 for x1, x2 in zip(x, y)]))


class KMeans():
    def __init__(self, X, labels, k_clusters=3, distance_metric='euclidean'):
        if not labels != list(set(labels)):
            raise ValueError("Labels must be unique!")

        self.X = X
        self.labels = labels
        self.map = dict(zip(labels, X))
        self.k_clusters = k_clusters
        self.distance_metric = euclidean

    def _closest(self, point, centroids, dfunc):
        """ Determine the closest centroid for a given point using the specified distance function """
        nearest = 0
        nearest_dist = dfunc(centroids[0], point)
        for i in range(1, len(centroids)):
            dist = dfunc(centroids[i], point)
            if dist < nearest_dist:
                nearest = i
                nearest_dist = dist
        return nearest

    def _find_closest_centroids(self, X, centroids, dfunc):
        cdict = defaultdict(list)
        for x in X:
            nearest = self._closest(x, centroids, dfunc)
            cdict[nearest].append(x)

        return cdict

    def _adjust_centroids(self, cdict):
        new_centroids = [np.mean(v, axis=0).tolist() for k, v in cdict.items()]
        return new_centroids

    def _compute_wcss(self, centroids, data, dfunc):
        wcss_total = 0.0
        for x in data:
            c = self._closest(x, centroids, dfunc)
            wcss_total += dfunc(centroids[c], x) ** 2
        return wcss_total

    def cluster(self):
        initial_centroids = random.sample(self.X, self.k_clusters)
        cdict = self._find_closest_centroids(self.X, initial_centroids, dfunc=self.distance_metric)
        new_centroids = self._adjust_centroids(cdict)

        while new_centroids != initial_centroids:
            initial_centroids = new_centroids
            cdict = self._find_closest_centroids(self.X, initial_centroids, dfunc=self.distance_metric)
            new_centroids = self._adjust_centroids(cdict)

        self.cdict = cdict
        self.centroids = new_centroids

    def _get_key(self, val):
        for k, v in self.map.items():
            if val == v:
                return k

    def _replace_cluster_with_label(self):
        new_cdict = defaultdict(list)
        for cluster, values in self.cdict.items():
            for idx, val in enumerate(values):
                label = self._get_key(val)
                new_cdict[cluster].append(label)
        return new_cdict

    @property
    def clusters(self):
        labeled_cdict = self._replace_cluster_with_label()
        for k, vals in labeled_cdict.items():
            print(f'Cluster {k}:')
            for item in vals:
                print(f'\t{item}')

    def visualize(self, label=False):
        fig, ax = plt.subplots()
        for cluster, vals in self.cdict.items():
            ax.scatter(x=[val[0] for val in vals], y=[val[1] for val in vals], label=cluster)
        # if label:
        #     for label, vals in zip(self.labels, self.cdict.values()):
        #         ax.annotate(label, (vals[]))
        fig.show()

    def visualize_wcss(self, kmin=1, kmax=10):
        """ Plot wcss as a function of k """
        wvals = []
        for k_cluster in range(kmin, kmax+1):
            k = KMeans(X=self.X, labels=self.labels, k_clusters=k_cluster)
            k.cluster()
            wcss = self._compute_wcss(k.centroids, k.X, dfunc=euclidean)
            wvals.append(wcss)

        plt.figure(figsize=(8, 8))
        plt.scatter(range(kmin, kmax+1), wvals, marker="X", c='r')
        plt.plot(range(kmin, kmax+1), wvals)
        plt.xlabel("K-Clusters")
        plt.ylabel("WCSS")
        plt.title("Finding optimal k-clusters using WCSS")
        plt.show()
