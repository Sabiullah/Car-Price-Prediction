import streamlit as st
import pandas as pd
import pickle

def load_model():
    # Load the model from pickle file
    with open('carPrice_pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_pipeline():
    # Load the pipeline from pickle file
    with open('CarPipeline_pkl', 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

def main():
    # Title of the app
    st.title('Car Price Prediction')

    # Load model and pipeline
    model = load_model()
    pipeline = load_pipeline()

    # Add dropdown for 'bt'
    bt_values = ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Pickup Trucks', 'Minivans', 'Coupe']
    bt_encoded_values = [1, 6, 5, 2, 4, 3, 0]
    bt_type = st.selectbox('bt', bt_values, index=bt_encoded_values.index(1))

    fuel_values = ['Diesel', 'Petrol', 'LPG', 'CNG']
    fuel_encoded_values = [1, 3, 2, 0]
    fuel_type = st.selectbox('Fuel Type', fuel_values, index=fuel_encoded_values.index(1))

    trans_values = ['Manual' 'Automatic']
    trans_encoded_values = [1, 0]
    trans_type = st.selectbox('Transmission', trans_values, index=trans_encoded_values.index(1))

    # Add dropdown for 'ownerNo'
    owner_no = st.selectbox('Owner Number', list(range(1, 6)), index=0)

    # Add dropdown for 'Registration Year'
    reg_year = st.selectbox('Registration Year', list(range(2002, 2023)), index=0)

    # Add input for 'Kms Driven'
    kms_driven = st.number_input('Kms Driven')

    # Add input for 'Mileage'
    mileage = st.number_input('Mileage', format='%f')

    # Add dropdown for 'Fuel Type'
    # fuel_type = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'LPG', 'CNG'])

    # Add dropdown for 'Transmission'
    # transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

    # Create feature vector for prediction
    features = pd.DataFrame({
        'bt': [bt_encoded_values[bt_values.index(bt_type)]],  # Use encoded value directly
        'ownerNo': [owner_no],
        'Registration Year': [reg_year],
        'Fuel Type': [fuel_encoded_values[fuel_values.index(fuel_type)]],
        'Kms Driven': [kms_driven],
        'Transmission': [trans_encoded_values[trans_values.index(trans_type)]],
        'Mileage': [mileage]
    })

    # Debugging: Print the input features before transformation
    # st.write('Input Features (Before Transformation):')
    # st.write(features)

    # Make prediction using the loaded model and pipeline
    transformed_features = pipeline.transform(features)

    # Debugging: Print the transformed features
    # st.write('Transformed Features:')
    # st.write(transformed_features)

    prediction = model.predict(transformed_features)

    # Display prediction
    st.write('Predicted Price:', round(prediction[0]))

if __name__ == '__main__':
    main()
