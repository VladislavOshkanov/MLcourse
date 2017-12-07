import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt



size = 100000

X = 5 * np.random.random_sample((size, 10))
normal = np.random.normal(0, 2, size)
u = np.random.random_sample((5,))
y = []
for i in range (0, size):
    sum = 0
    for j in range (0, 5):
        sum = sum + X[i,0] ** (j+1) * u[j]
    y.append( sum + normal [i] )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = np.array(X_train)
X_test = np.array(X_test)
mae_test_res = []
mae_train_res = []

for i in range (1, 11):
    lr = linear_model.LinearRegression()
    X_train_first_columns = X_train[:, :i]
    X_test_first_columns = X_test[:, :i]
    lr.fit(X_train_first_columns, y_train)

    y_predicted_train = lr.predict(X_train_first_columns)
    y_predicted_test = lr.predict(X_test_first_columns)

    mae_train_res.append(mae(y_train, y_predicted_train))
    mae_test_res.append(mae(y_test, y_predicted_test))





print (mae_train_res)
print (mae_test_res)

x = np.arange(1, 11, 1);

lines = plt.plot(x, mae_train_res,'b', x , mae_test_res, 'g')
lines[0].set_antialiased(True)
plt.xticks(x)
plt.grid(True)
plt.show()
