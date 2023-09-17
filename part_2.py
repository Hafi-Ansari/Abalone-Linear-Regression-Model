from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

#load and preprocess the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
df = pd.read_csv(url, names=column_names)

#remove duplicates and null values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

#one-hot encode the 'Sex' column
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

#split the data into training and test sets
X = df.drop('Rings', axis=1)
y = df['Rings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

#make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#calculate and print the Mean Squared Error
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Mean Squared Error on the training set: {train_mse}")
print(f"Mean Squared Error on the test set: {test_mse}")

#print the model parameters
print("Final Weights:", model.coef_)
print("Final Bias:", model.intercept_)
