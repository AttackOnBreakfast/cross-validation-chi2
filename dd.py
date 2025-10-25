import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# 1) Data: linear teacher with small noise
n = 200          # samples
d = 50           # truly informative features
sigma = 0.1      # noise level

X_base = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
y = X_base @ w_true + rng.normal(scale=sigma, size=n)

# 2) Sweep model size p across and beyond n
p_values_left  = list(range(5, d+1, 5))
p_values_right = list(range(d+5, 3*n+1, 20))   # go well beyond n
p_values = p_values_left + p_values_right

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_mse_means = []
train_mse = []

for p in p_values:
    if p <= d:
        Xp = X_base[:, :p]
    else:
        extra = rng.normal(size=(n, p - d))
        Xp = np.concatenate([X_base, extra], axis=1)

    lr = LinearRegression(fit_intercept=False)

    # 3–4) K-fold CV MSE
    neg_mse = cross_val_score(lr, Xp, y, scoring="neg_mean_squared_error", cv=cv)
    cv_mse_means.append(-neg_mse.mean())

    # 5) Training MSE (optional)
    lr.fit(Xp, y)
    y_hat = lr.predict(Xp)
    train_mse.append(np.mean((y - y_hat)**2))

# --- Plot CV MSE (Double Descent curve) ---
plt.figure()
plt.title("Double descent with K-fold CV (Linear Regression)")
plt.xlabel("Number of features p")
plt.ylabel("CV MSE")
plt.plot(p_values, cv_mse_means, marker="o")
plt.axvline(n, linestyle="--")  # interpolation threshold
plt.tight_layout()
plt.show()

# --- Plot Training MSE (separate figure) ---
plt.figure()
plt.title("Training MSE vs model size")
plt.xlabel("Number of features p")
plt.ylabel("Training MSE")
plt.plot(p_values, train_mse, marker="o")
plt.axvline(n, linestyle="--")
plt.tight_layout()
plt.show()