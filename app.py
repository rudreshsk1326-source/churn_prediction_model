import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title='Churn Prediction', page_icon='🏦', layout='centered')

st.title('🏦 Customer Churn Prediction')
st.markdown('Fill in the customer details below and click **Predict** to see the result.')
st.divider()

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92)
    balance = st.number_input('💰 Balance', min_value=0.0)
    credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=900, value=600)

with col2:
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0)
    tenure = st.slider('📅 Tenure (Years)', 0, 10)
    num_of_products = st.slider('🛒 Number of Products', 1, 4)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

st.divider()

if st.button('🔍 Predict Churn', use_container_width=True):
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)
    input_data = input_data[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_data)

    prediction_proba = float(model.predict(input_scaled)[0][0])
    churn = prediction_proba > 0.5

    st.divider()
    st.subheader('📋 Prediction Result')

    r1, r2, r3 = st.columns(3)
    r1.metric('🎯 Churn Probability', f'{prediction_proba:.2%}')
    r2.metric('📈 Model Accuracy', '80%')
    r3.metric('🔮 Prediction', '⚠️ Churn' if churn else '✅ Retain')

    st.markdown('**Churn Risk Level**')
    st.progress(prediction_proba)

    if churn:
        st.error(f'⚠️ This customer is **likely to churn** with a probability of **{prediction_proba:.2%}**. Consider taking retention actions.')
    else:
        st.success(f'✅ This customer is **not likely to churn** with a churn probability of only **{prediction_proba:.2%}**.')
