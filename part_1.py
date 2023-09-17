"""Python File For Linear Regression using Gradient Descent"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""DATAFRAME SETUP"""

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]

df = pd.read_csv(url, names=column_names)
df.head()



"""PRE-PROCESSING DATA"""

#check for null values
null_values = df.isnull().sum()
#print(null_values) -> found none so commented out 

#redundant rows
df.drop_duplicates(inplace=True)

#one-hot encoding the 'Sex' column to get a numerical value instead of categorical
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

#check for correlation with the target variable
correlations = df.corr()['Rings'].sort_values()
#print(correlations) -> correlation found to be strong for most/all variables so commented out 



"""SPLIT DATASET (80/20)"""

X = df.drop('Rings', axis=1)
y = df['Rings']
#splitting data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



"""IMPLEMENTING GRADIENT DESCENT TO MAKE A LINEAR REGRESSION MODEL"""

num_features = 9
weights = np.zeros(num_features)
bias = 0

#step size will be small as we go towards the convex line minima
learning_rate = 0.01
#decent sized iteration number
num_iterations = 1000

#list to store the MSE with every iteration
train_errors = []

#Gradient Descent Magic

# Open the log file in append mode
with open("log.txt", "a") as log_file:
    # Gradient Descent Magic
    for i in range(num_iterations):
        # numpy dot function takes the dot product of weights of each feature
        y_pred = np.dot(X_train, weights) + bias

        # calculating error(s)
        error = y_train - y_pred
        # np.mean -> Mean Squared Error (mse)
        mean_squared_error = np.mean(np.square(error))
        # at every iteration we will append the current MSE to the list
        train_errors.append(mean_squared_error)

        # calculating gradients
        weight_gradient = -(2/len(X_train)) * np.dot(X_train.T, error)
        bias_gradient = -(2/len(X_train)) * np.sum(error)

        # updating weights and bias
        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

        # prediction on test set
        y_test_pred = np.dot(X_test, weights) + bias
        test_error = y_test - y_test_pred
        test_mse = np.mean(np.square(test_error))

        # logging error values
        if i % 50 == 0:
            # Write the log details to the file
            log_file.write(f"TRAINING SET:\tIteration {i}, Mean Squared Error: {mean_squared_error}\n")
            log_file.write(f"TESTING SET:\tIteration {i}, Mean Squared Error: {test_mse}\n")
            log_file.write(f"Weights: {weights}\n\n")
print("Training and Testing Data parameters, iterations, and MSE have been logged to log.txt\n")

# Store the final weights and bias
final_weights = weights
final_bias = bias

print("Final Weights:")
list(map(lambda x: print(x), final_weights))

print(f"\nFinal Bias: {final_bias}\n\n")

#using the final weights and bias to make predictions on the test set
y_test_pred = np.dot(X_test, final_weights) + final_bias

#calculating the error for the test set
test_error = y_test - y_test_pred

#calculating the Mean Squared Error (MSE) for the test set
test_mse = np.mean(np.square(test_error))

print(f"\nMEAN SQUARED ERROR ON THE TEST SET USING FINAL WEIGHTS FROM TRAINING SET: {test_mse}")

print(f"\n\nI believe I have found the best solution as the MSE for both training set and testing set stabilizes at 6.8.")