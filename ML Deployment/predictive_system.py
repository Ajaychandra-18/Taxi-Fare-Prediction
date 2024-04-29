import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import pickle
#loading the saved model
loaded_model = pickle.load(open('E:/ML Deployment/trained_model.sav','rb'))
df = pd.read_csv('E:/ML Deployment/train.csv')

# Feature Engineering Function
def feature_engineering(df):
    # Calculate distance based on distance_traveled
    df['distance'] = df['distance_traveled']

    # Extract additional features from fare and tip
    df['fare_per_distance'] = df['fare'] / df['distance']
    df['tip_percentage'] = (df['tip'] / df['total_fare']) * 100

    # Drop unnecessary columns after feature engineering
    df.drop(['distance_traveled'], axis=1, inplace=True)

    return df

# Run the feature engineering function
df = feature_engineering(df)

# Now the distance column is in the DataFrame
X = df[['trip_duration', 'distance', 'num_of_passengers', 'fare', 'tip', 'miscellaneous_fees', 'surge_applied']]
y = df['total_fare']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)
