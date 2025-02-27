import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
            It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session
        print(X)
        matriz = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X),X)),np.transpose(X)),y)
        matriz = np.transpose(matriz)
        self.intercept = matriz[0]
        self.coefficients = matriz[1:]


    def fit_gradient_descent(self, X, valores_de_verdad, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.
   
        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.
 
        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        n = len(valores_de_verdad)
        self.intercept = np.random.rand() * 0.01
        self.coefficients = np.random.rand(X.shape[1] - 1) * 0.01
        for epoch in range(iterations):
            y_pred = self.intercept + X[:, 1:].dot(self.coefficients)
            error = y_pred - valores_de_verdad
            self.intercept -= learning_rate*error.mean()
            self.coefficients -= learning_rate * X[:, 1:].T.dot(error)/n
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: MSE = {error**2/n}")



    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        if np.ndim(X) == 1:
            predictions = self.intercept + self.coefficients*X
        else:
            # TODO: Predict when X is more than one variable
            predictions = self.intercept + np.matmul(X,np.transpose(self.coefficients))
        return predictions



def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    rss = 0
    tss = 0
    mean_abs_error = 0

    for i in range(len(y_true)):
        rss += (y_true[i]-y_pred[i])**(2)
        tss += (y_true[i]-y_true.mean())**(2)
        mean_abs_error += abs(y_true[i]-y_pred[i])
    r_squared = 1-(rss/tss)
    root_mean_se = (rss/len(y_true))**(0.5)
    mae = mean_abs_error/len(y_true)

    return {"R2": r_squared, "RMSE": root_mean_se, "MAE": mae}

def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for col in sorted(categorical_indices, reverse=True):
        col_data = X_transformed[:, col]
        uniques = []
        for v in col_data:
            if v not in uniques:
                uniques.append(v)
        mapping = {v: i for i, v in enumerate(uniques)}
        one_hot = np.eye(len(uniques), dtype=int)[[mapping[v] for v in col_data]]
        if drop_first:
            one_hot = one_hot[:, 1:]
        X_transformed = np.hstack((X_transformed[:, :col], one_hot, X_transformed[:, col+1:]))
    return X_transformed
