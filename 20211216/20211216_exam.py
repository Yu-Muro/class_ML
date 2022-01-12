# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.datasets
import sklearn.decomposition
import mlxtend.plotting

np.random.seed( 1 )
data, label = sklearn.datasets.make_moons( n_samples=200, shuffle=True, noise=0.15, random_state = 1 )
A = np.array([[10.0*np.cos(np.pi/4), np.sin(np.pi/4)],[-10.0*np.sin(np.pi/4), np.cos(np.pi/4)]])
data = np.dot( data, A ) + [100.0, -10.0]


for i in range(100, 111):
    plt.figure()
    model = sklearn.svm.SVC(kernel="rbf",gamma=i / 10)
    model.fit(data, label)
    mlxtend.plotting.plot_decision_regions(data, label, model)
    plt.savefig("data{}.png".format(i))

