from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
import numpy as np


boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=4648)

rf = RandomForestRegressor(n_estimators=30, n_jobs=4)
rf.fit (X_train, y_train)
y_predicted_rf = rf.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_predicted_lr = lr.predict(X_test)


print (mae(y_test, y_predicted_rf))
print (mae(y_test, y_predicted_lr))

importances = rf.feature_importances_
print (np.argmax(importances))

mae_res = []
for i in range (0,12):
    a = np.array(X_train)
    b = np.array (X_test)
    X_train_without_column = np.concatenate((a[:,0:i],a[:, i+1:13]), axis=1)
    X_test_without_column = np.concatenate((b[:,0:i],b[:, i+1:13]), axis=1)
    lr = LinearRegression()
    lr.fit(X_train_without_column, y_train)
    y_predicted_without_column = lr.predict(X_test_without_column)
    mae_res.append(mae(y_test, y_predicted_without_column))

print (np.argmax(mae_res))
