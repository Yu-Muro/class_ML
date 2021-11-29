import os
import sys
import config
imp = __import__("20211129_exam")

n = int(sys.argv[1])
imp.linear_regression(n)

png = os.listdir("figure")
if "model_a_scatter.png" not in png:
    imp.make_scatter(config.ya, "a")
if "model_b_scatter.png" not in png:
    imp.make_scatter(config.yb, "b")