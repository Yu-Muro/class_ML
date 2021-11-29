def mse(n, x, y, model):
    s = 0
    for i in range(len(x)):
        tilde = 0
        for j in range(n):
            tilde += model.coef_[j] * (x[i]**(j+1))
        tilde += model.intercept_
        epsilon_a = (tilde - y[i])**2
        s += epsilon_a
    return s / len(x)