import sys
imp = __import__("20211129_exam")

x, y = int(sys.argv[1]), int(sys.argv[2])
for n in range(x, y):
    imp.linear_regression(n)