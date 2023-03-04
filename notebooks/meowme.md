#### Define the data points

```R
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]
```

#### Calculate means of X and Y

```R
mean_X = sum(X) / len(X)
mean_Y = sum(Y) / len(Y)
```

#### Calculate variance of X and covariance of X and Y
```R
var_X = sum([(x - mean_X)**2 for x in X]) / len(X)
cov_XY = sum([(x - mean_X) * (y - mean_Y) for x, y in zip(X, Y)]) / len(X)
```

#### Calculate the slope and y-intercept of the regression line

```R
slope = cov_XY / var_X
y_intercept = mean_Y - slope * mean_X
```

#### Define a function that predicts the Y values for a given X value

```R
def predict_Y(x):
    return slope * x + y_intercept
```

# Test the linear regression model by predicting Y for a given X value

```R
x_test = 6
y_pred = predict_Y(x_test)
print("For X = {}, predicted Y value is {}".format(x_test, y_pred))
```
