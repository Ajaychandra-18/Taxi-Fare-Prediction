import pickle
import streamlit as st
import numpy as np

# Load the trained model
pickle_in = open('E:/ML Deployment/Regression.pkl', 'rb')
model = pickle.load(pickle_in)

# Creating a function for prediction
def prediction(trip_duration, distance, num_of_passengers, fare, tip, miscellaneous_fees, surge_applied):
    input_data = np.array([[trip_duration, distance, num_of_passengers, fare, tip, miscellaneous_fees, surge_applied]])
    return model.predict(input_data)

# Streamlit UI
def main():
    st.title('Taxi Fare Prediction')

    # Input fields
    trip_duration = st.number_input('Trip Duration in minutes')
    distance_traveled = st.number_input('Distance in kms')
    
    # Dropdown menu for number of passengers
    num_of_passengers = st.selectbox('Number of Passengers', list(range(1, 7)), index=0)
    
    fare = st.number_input('Fare')
    tip = st.number_input('How much tip do you want to give?')
    miscellaneous_fees = st.number_input('Miscellaneous Fees')
    
    # Surge Applied dropdown
    surge_options = ('Day', 'Night')
    surge_applied = st.selectbox('Are you traveling at night or during the day?', surge_options)

    # Convert surge_applied to 1 for Night and 0 for Day
    surge_applied_value = 1 if surge_applied == 'Night' else 0

    # Predict total fare
    if st.button('Predict Total Fare'):
        total_fare = prediction(trip_duration, distance_traveled, num_of_passengers, fare, tip, miscellaneous_fees, surge_applied_value)
        st.write(f'Predicted total fare: ${total_fare[0]:.2f}')

if __name__ == '__main__':
    main()
