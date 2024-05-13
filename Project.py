import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

def perform_linear_regression():
    # Load datasets
    pesticides_df = pd.read_csv('pesticides.csv')
    rainfall_df = pd.read_csv('rainfall.csv')
    temp_df = pd.read_csv('temp.csv')
    yield_df = pd.read_csv('yield.csv')

    # Merge datasets based on 'Year' and 'Area'
    merged_df = pd.merge(pesticides_df, rainfall_df, on=['Year', 'Area'], how='inner')
    merged_df = pd.merge(merged_df, temp_df, left_on=['Year', 'Area'], right_on=['year', 'country'], how='inner')
    merged_df = pd.merge(merged_df, yield_df, left_on=['Year', 'Area'], right_on=['Year', 'Area'], how='inner')

    # Clean data
    merged_df.replace('..', np.nan, inplace=True)
    merged_df.dropna(inplace=True)

    # Encode categorical data
    label_encoder = LabelEncoder()
    merged_df['Area'] = label_encoder.fit_transform(merged_df['Area'])

    # # Prepare features and target
    # X = merged_df[['Year', 'Area', 'Value_x', 'average_rain_fall_mm_per_year', 'avg_temp']]
    # y = merged_df['Value_y']

   # Prepare features and target
    X = merged_df[['Year', 'Area', 'Value_x', 'average_rain_fall_mm_per_year', 'avg_temp']]
    X.columns = ['Year', 'Area', 'Pesticides', 'Rainfall', 'Temperature']
    y = merged_df['Value_y']


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # print("Mean Squared Error (MSE):", mse)
    # print("Mean Absolute Error (MAE):", mae)
    import warnings
    warnings.filterwarnings('ignore')


    # Take user input for parameters
    while True:
        year = int(input("Enter the year: "))
        area = label_encoder.transform([input("Enter the area: ")])[0]
        pesticides = float(input("Enter the pesticides value: "))
        rainfall = float(input("Enter the average rainfall (mm/year): "))
        temperature = float(input("Enter the average temperature (°C): "))

        # Make prediction for user input
        prediction = model.predict([[year, area, pesticides, rainfall, temperature]])
        print("Prediction:", prediction[0])
        
        cont_input=input("do you want to continue?: (yes/no)")
        if cont_input.lower()!='yes':
            break

# Call the function to perform linear regression and make predictions
perform_linear_regression()