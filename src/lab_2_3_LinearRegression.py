# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # TODO: Train linear regression model with only one coefficient
        y_media = np.mean(y)
        x_media = np.mean(X)
        n = len(X)
    
        primer_sum = 0
        segundo_sum = 0
        tercer_sum = 0
        cuarta_sum = 0

        for i in range(0,n):
            primer_sum += X[i]*y[i]
            tercer_sum += X[i]*X[i]
            for j in range(0,n):
                segundo_sum += X[i]*y[j]
                cuarta_sum += X[i]*X[j]
        Sxy = (primer_sum)-(segundo_sum/n)
        S2x = (tercer_sum)-(cuarta_sum/n)
    
        self.coefficients = Sxy/S2x
        self.intercept = y_media - x_media*self.coefficients



    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # TODO: Train linear regression model with multiple coefficients

        nueva_columna = np.array([1 for i in range(len(X))])
        X = np.insert(X,0,nueva_columna,axis=1)
        for linea in X:
            print(linea)
        matriz = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),y)
        matriz = np.transpose(matriz)
        self.intercept = matriz[0]
        self.coefficients = matriz[1:]

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
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
    # R^2 Score
    media_ytrue = np.mean(y_true)
    TSS = 0
    RSS = 0
    N = len(y_true)
    MAE = 0
    for i in range(0,N):
        TSS += (y_true[i]-media_ytrue)**2
        RSS += (y_true[i]-y_pred[i])**2
        MAE += abs(y_true[i]-y_pred[i])
    # TODO: Calculate R^2
    r_squared = 1- RSS/TSS

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = (RSS/N)**(1/2)

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = MAE/N

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    """Compares a custom linear regression model with scikit-learn's LinearRegression.

    Args:
        x (numpy.ndarray): The input feature data (1D array).
        y (numpy.ndarray): The target values (1D array).
        linreg (object): An instance of a custom linear regression model. Must have
            attributes `coefficients` and `intercept`.

    Returns:
        dict: A dictionary containing the coefficients and intercepts of both the
            custom model and the scikit-learn model. Keys are:
            - "custom_coefficient": Coefficient of the custom model.
            - "custom_intercept": Intercept of the custom model.
            - "sklearn_coefficient": Coefficient of the scikit-learn model.
            - "sklearn_intercept": Intercept of the scikit-learn model.
    """
    ### Compare your model with sklearn linear regression model
    # TODO : Import Linear regression from sklearn
    from sklearn.linear_model import LinearRegression
    # Assuming your data is stored in x and y
    # TODO : Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features

    x_reshaped = x.reshape(-1, 1)

    print(x_reshaped)
    # Create and train the scikit-learn model
    # TODO : Train the LinearRegression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    """Loads Anscombe's quartet, fits custom linear regression models, and evaluates performance.

    Returns:
        tuple: A tuple containing:
            - anscombe (pandas.DataFrame): The Anscombe's quartet dataset.
            - datasets (list): A list of unique dataset identifiers in Anscombe's quartet.
            - models (dict): A dictionary where keys are dataset identifiers and values
              are the fitted custom linear regression models.
            - results (dict): A dictionary containing evaluation metrics (R2, RMSE, MAE)
              for each dataset.
    """
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    # TODO: Construct an array that contains, for each entry, the identifier of each dataset
    datasets = None

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        # TODO
        data = None

        # Create a linear regression model
        # TODO
        model = None

        # Fit the model
        # TODO
        X = None  # Predictor, make it 1D for your custom model
        y = None  # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        # TODO
        y_pred = None

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    return results


# Go to the notebook to visualize the results
