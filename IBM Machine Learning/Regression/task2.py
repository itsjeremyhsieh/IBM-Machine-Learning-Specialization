# %%
from tqdm import tqdm
import skillsnetwork
import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, load_wine

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# %%
df = pd.read_csv('cleaned_data.csv')
print(df.info())

df = pd.get_dummies(df)
df = df.sort_values('Sleep duration')

X = df.drop(columns=['Sleep efficiency', 'ID'])
Y = df['Sleep efficiency'].copy()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 42)
print(y_test.info())
# %%

lr = LinearRegression()
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
lr.score(X_train,y_train)
lr.score(X_test,y_test)
print("R^2 on training data:", lr.score(X_train,y_train))
print("R^2 on testing data ", lr.score(X_test,y_test))
mse = mean_squared_error(y_test, predicted)
print("Mean Squared Error: ", mse)
import seaborn as sns
ax = plt.axes()
ax.scatter(y_test, predicted, alpha=.5)
plt.plot([0,1],[0,1], transform=ax.transAxes, color='r')

plt.xlabel("Test")
plt.ylabel("Predicted")
plt.show()

# %%
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

lm = LinearRegression()
lm.fit(X_train_poly, y_train)
poly_predicted = lm.predict(X_test_poly)
print("R^2 on training data:", lm.score(X_train_poly, y_train))
print("R^2 on testing data:", lm.score(X_test_poly,y_test))
mse = mean_squared_error(y_test, poly_predicted)
print("Mean Squared Error: ", mse)

# %%
print(y_test.size)
print(poly_predicted.size)

# %%

import seaborn as sns
ax = plt.axes()
ax.scatter(y_test, poly_predicted, alpha=.5)
plt.plot([0,1],[0,1], transform=ax.transAxes, color='r')

plt.xlabel("Test")
plt.ylabel("Predicted")
plt.show()
# %%
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.5)
ridge.fit(X_train_poly, y_train)
y_pred = ridge.predict(X_test_poly)

print("R^2 on training data:", lm.score(X_train_poly, y_train))
print("R^2 on testing data:", lm.score(X_test_poly,y_test))
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

import seaborn as sns
ax = plt.axes()
ax.scatter(y_test, y_pred, alpha=.5)

plt.plot([0,1],[0,1], transform=ax.transAxes, color='r')
plt.xlabel("Test")
plt.ylabel("Predicted")
plt.show()
# %%
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.5)
lasso.fit(X_train_poly, y_train)
y_pred = lasso.predict(X_test_poly)

print("R^2 on training data:", lm.score(X_train_poly, y_train))
print("R^2 on testing data:", lm.score(X_test_poly,y_test))
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

import seaborn as sns
ax = plt.axes()
ax.scatter(y_test, y_pred, alpha=.5)

plt.plot([0,1],[0,1], transform=ax.transAxes, color='r')
plt.xlabel("Test")
plt.ylabel("Predicted")
plt.show()
# %%
coefficients = pd.DataFrame()
coefficients['linear regression'] = lr.coef_.ravel()
coefficients['ridge regression'] = ridge.coef_.ravel()
coefficients['lasso regression'] = lasso.coef_.ravel()
coefficients = coefficients.applymap(abs)

coefficients.describe()  # Huge difference in scale between non-regularized vs regularized regression
# %%
