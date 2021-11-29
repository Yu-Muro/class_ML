# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

import config
from mean_squared_error import mse


def linear_regression(n):
    print("=======================================================================================")

    X = np.array([config.x**(i + 1) for i in range(n)]).T
    ya, yb = config.ya.T, config.yb.T

    model_A = sklearn.linear_model.LinearRegression()
    model_A.fit(X, ya)
    print( 'coefficient_A:', model_A.coef_, ' intercept_A', model_A.intercept_ )
    Ea = mse(n, config.x, ya, model_A)
    print("mean squared error of a {:.10f}".format(Ea))
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    model_B = sklearn.linear_model.LinearRegression()
    model_B.fit(X, yb)
    print( 'coefficient_B:', model_B.coef_, ' intercept_B', model_B.intercept_ )
    Eb = mse(n, config.x, yb, model_B)
    print("mean squared error of b {:.10f}".format(Eb))

    folder = os.listdir()
    if "figure" not in folder:
        os.mkdir("figure")

    x_pred = np.arange(-1.0, 1.001, 0.001)
    X_pred = np.array([x_pred**(i+1) for i in range(n)]).T
    ya_pred = model_A.predict( X_pred )
    yb_pred = model_B.predict( X_pred )

    plt.figure()
    plt.plot(x_pred, ya_pred, color='red')
    plt.scatter(config.x, ya)
    plt.ylim([-2.0, 2.0])
    plt.savefig("figure/model_a_n={}.png".format(n))

    plt.figure()
    plt.plot(x_pred, yb_pred, color='red')
    plt.scatter(config.x, yb)
    plt.ylim([-2.0, 2.0])
    plt.savefig("figure/model_b_n={}.png".format(n))


def make_scatter(y, n):
    y = y.T
    plt.figure()
    plt.scatter(config.x, y)
    plt.ylim([-2.0, 2.0])
    plt.savefig("figure/model_{}_scatter.png".format(n))