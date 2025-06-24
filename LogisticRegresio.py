# 📦 Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 🧪 Create sample classification data (binary classes 0 and 1)
X, y = make_classification(n_samples=100,    # 100 data points
                           n_features=2,     # 2 input features (so we can plot)
                           n_informative=2,  # all features are useful
                           n_redundant=0,    # no unnecessary features
                           n_clusters_per_class=1,
                           random_state=42)

# ⚙️ Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# 📊 Predict for the first input point
print("Prediction for first sample (X[0]):", model.predict([X[0]]))
print("Prediction probabilities:", model.predict_proba([X[0]]))

# 🎨 Plot data points
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Logistic Regression Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 🌗 Plot the decision boundary
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(model.coef_[0][0] * x_vals + model.intercept_[0]) / model.coef_[0][1]
plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')

plt.legend()
plt.grid(True)
plt.show()
