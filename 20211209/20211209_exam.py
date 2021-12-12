# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.decomposition
import mlxtend.plotting

data_num = 300
np.random.seed(1)
class0_data = [6,1] + [3.1, 1] * np.random.randn(data_num//3,2)
class1_data = [-7,6] + [3.1, 1] * np.random.randn(data_num//3,2)
class2_data = [-3,-3] + [3.1, 1] * np.random.randn(data_num//3,2)
label = np.array([ k//100 for k in range(data_num) ] )
data = np.vstack( ( class0_data, class1_data, class2_data ) )

std_data = data.copy()
std_data[:,0] = ( data[:,0] - data[:,0].mean() ) / data[:,0].std()
std_data[:,1] = ( data[:,1] - data[:,1].mean() ) / data[:,1].std()

plt.figure()
model = sklearn.linear_model.LogisticRegression()
model.fit(data, label)
mlxtend.plotting.plot_decision_regions(data, label, model)
plt.savefig("raw_data.png")

plt.figure()
std_model = sklearn.linear_model.LogisticRegression()
std_model.fit(std_data, label)
mlxtend.plotting.plot_decision_regions(std_data, label, std_model)
plt.savefig("std_data.png")