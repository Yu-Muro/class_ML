# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

from mean_squared_error import mse

def linear_regression(n):
    print("=======================================================================================")
    x = np.arange( -1, +1.1, 0.2 )
    ya = np.array([-0.26114704, 0.04081065, 0.40065032, 0.65437830, 0.66576410, 0.36474044, -0.21419675, -0.89358835, -1.31461206, -0.89868233, 1.19094953])
    yb = np.array([ 0.06372203, -0.08154063, 0.29501597, 0.43978457, 0.83884562, -0.09556730, 0.13476560, -1.04582973, -1.25080424, -0.94855641, 1.48337112])

    X = np.array([x**(i + 1) for i in range(n)]).T
    ya, yb = ya.T, yb.T

    model_A = sklearn.linear_model.LinearRegression()
    model_A.fit(X, ya)
    print( 'coefficient_A:', model_A.coef_, ' intercept_A', model_A.intercept_ )
    Ea = mse(n, x, ya, model_A)
    print("mean squared error of a {:.10f}".format(Ea))
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    model_B = sklearn.linear_model.LinearRegression()
    model_B.fit(X, yb)
    print( 'coefficient_B:', model_B.coef_, ' intercept_B', model_B.intercept_ )
    Eb = mse(n, x, yb, model_B)
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
    plt.scatter(x, ya)
    plt.ylim([-2.0, 2.0])
    plt.savefig("figure/model_a_n={}.png".format(n))

    plt.figure()
    plt.plot(x_pred, yb_pred, color='red')
    plt.scatter(x, yb)
    plt.ylim([-2.0, 2.0])
    plt.savefig("figure/model_b_n={}.png".format(n))