import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("20211125_exam.csv")
for i in range(4):
    x = np.array(df["x{}".format(i+1)])
    y = np.array(df["y{}".format(i+1)])
    print("x{0}の平均値:{1}, y{0}の平均値:{3:.10f}, x{0}の分散:{2:.10f}, y{0}の分散:{4:.10f}"
        .format(i+1, x.mean(), x.var(), y.mean(), y.var()))
    print("共分散行列")
    print( np.cov( x, y, ddof=0 ) )
    x_z = ( x - x.mean() ) / x.std()
    y_z = ( y - y.mean() ) / y.std()
    print("相関行列")
    print( np.cov( x_z, y_z, ddof=0 ) )
    print("===========================================")
    plt.figure()
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.grid(color='gray')
    plt.xlabel("x{}".format(i+1))
    plt.ylabel("y{}".format(i+1))
    plt.scatter(x, y)
    plt.scatter(x.mean(), y.mean(), color='red', marker='D')
    plt.legend(["data", "mean"])
    plt.savefig("figure{}.png".format(i+1))
