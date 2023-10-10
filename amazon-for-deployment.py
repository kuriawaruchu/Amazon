# import streamlit as st
# import pandas as pd
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# # Load the training and test data
# training_data = pd.read_csv('https://raw.githubusercontent.com/kuriawaruchu/Amazon/main/training_data.csv')
# test_data = pd.read_csv('https://raw.githubusercontent.com/kuriawaruchu/Amazon/main/test_data.csv')

# # Specify SARIMA hyperparameters
# p = 1
# d = 0
# q = 1
# seasonal_p = 1
# seasonal_d = 0
# seasonal_q = 1
# s = 5

# # Fit the SARIMA model
# sarima_model = SARIMAX(training_data['Adj Close'], order=(p, d, q), seasonal_order=(seasonal_p, seasonal_d, seasonal_q, s))
# sarima_results = sarima_model.fit()

# # Make predictions for the test set
# test_predictions = sarima_results.predict(start=len(training_data), end=len(training_data) + len(test_data) - 1, dynamic=False)

# # Display the results
# st.title('Amazon Stock Price Predictions')
# st.line_chart(test_predictions)
# st.line_chart(test_data['Adj Close'])
# # st.legend()

import streamlit as st
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


main_data_diff_df = pd.read_csv('https://raw.githubusercontent.com/kuriawaruchu/Amazon/main/main_data_diff_df.csv')

# Split the DataFrame into x and y variables
X = main_data_diff_df.drop('Adj Close', axis=1)
y = main_data_diff_df['Adj Close']

# Split the x and y variables into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Define the SARIMA model function
def generate_sarima_predictions(start_date, end_date):
    # Specify SARIMA hyperparameters (you can adjust these as needed)
    p = 2
    d = 0
    q = 2
    seasonal_p = 2
    seasonal_d = 0
    seasonal_q = 2
    s = 5  # daily data

    # ... Your data loading and preprocessing code ...

    # Fit the SARIMA model
    sarima_model = sm.tsa.SARIMAX(y_train, order=(p, d, q), seasonal_order=(seasonal_p, seasonal_d, seasonal_q, s))
    sarima_results = sarima_model.fit()

    # Generate predictions for the specified date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    predictions = sarima_results.predict(start=len(y_train), end=len(y_train) + len(date_range) - 1, dynamic=False)

    # Create a DataFrame with the predictions and date range
    predictions_df = pd.DataFrame({'Datetime': date_range, 'Predicted_Adj_Close': predictions})

    return predictions_df

# Create a Streamlit web app
def main():
    st.title('Amazon Stock Price Predictor')

    # User input for date range
    start_date = st.date_input('Select Start Date', datetime(2023, 10, 7))
    end_date = st.date_input('Select End Date', datetime(2024, 1, 9))

    if start_date < end_date:
        st.write(f"Generating predictions for {start_date} to {end_date}...")

        # Call the SARIMA model function
        predictions_df = generate_sarima_predictions(start_date, end_date)

        # Display predictions
        st.write(predictions_df)

        # Optionally save predictions to a CSV file
        if st.button('Save Predictions'):
            predictions_df.to_csv('sarima_predictions.csv', index=False)
            st.success('Predictions saved to sarima_predictions.csv')
    else:
        st.error('Please select a valid date range.')

if __name__ == '__main__':
    main()
