import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Tries to load locally first.
    # Make sure your new saved model (e.g., 'Random_Forest_Tuned_model.joblib') is in the same folder.
    model_path = 'Random_Forest_Tuned_model.joblib'

    # Fallback for Colab absolute path
    if not os.path.exists(model_path):
        model_path = '/content/Random_Forest_Tuned_model.joblib'

    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Handle missing model file
if model is None:
    st.error("⚠️ Model file not found.")
    st.info("Please ensure your trained model (e.g., 'Random_Forest_Tuned_model.joblib') is uploaded to your Colab session.")
    st.stop()

# --- 2. UI LAYOUT ---
st.title('Soil Nail Wall Stability Predictor')
st.markdown("### Predict Factor of Safety (FoS)")
st.write("Adjust the design parameters below to calculate the global stability of the reinforced excavation.")

# Create tabs to organize inputs cleanly
tab1, tab2, tab3 = st.tabs(["Excavation Geometry", "Nail Configuration", "Soil Properties"])

with tab1:
    st.subheader("Geometry Parameters")
    col1, col2 = st.columns(2)
    with col1:
        # Based on dataset range: 10 to 15
        height = st.slider('Excavation Height (H) [m]', min_value=5.0, max_value=15.0, value=10.0, step=0.5)
        # Based on dataset range: 45 to 90
        slope_angle = st.slider('Slope Angle [deg]', min_value=45, max_value=90, value=75, step=1)
    with col2:
        # Based on dataset range: 0 to 20
        back_slope_angle = st.slider('Back Slope Angle [deg]', min_value=0, max_value=20, value=10, step=1)

with tab2:
    st.subheader("Soil Nail Design")
    col3, col4 = st.columns(2)
    with col3:
        # Based on dataset range: 0.5 to 1.2
        l_h_ratio = st.slider('Length/Height Ratio (L/H)', min_value=0.5, max_value=1.2, value=0.6, step=0.1)
        # Based on dataset range: 10 to 20
        nail_inclination = st.slider('Nail Inclination [deg]', min_value=0, max_value=35, value=15, step=1)
    with col4:
        # Based on dataset range: 1.25 to 2.0
        spacing_vertical = st.slider('Vertical Spacing [m]', min_value=1.25, max_value=2.00, value=1.96, step=0.01)

with tab3:
    st.subheader("Geotechnical Parameters")
    col5, col6 = st.columns(2)
    with col5:
        # Static at 10 in training data
        cohesion = st.slider('Cohesion (c) [kPa]', min_value=5, max_value=195, value=10, step=5)
    with col6:
        # Static at 35 in training data
        friction_angle = st.slider('Friction Angle (φ) [deg]', min_value=10, max_value=35, value=35, step=5)

st.markdown("---")

# --- 3. PREDICTION LOGIC ---
if st.button('Predict Factor of Safety', type="primary"):

    # Construct DataFrame EXACTLY matching your new input vector keys
    # The order and spelling must perfectly match the training columns
    input_data = pd.DataFrame({
        'BackSlopeAngle': [back_slope_angle],
        'SlopeAngle': [slope_angle],
        'NailInclination': [nail_inclination],
        'L/H': [l_h_ratio],
        'SpacingVertical': [spacing_vertical],
        'Height': [height],
        'Cohesion': [cohesion],
        'FrictionAngle': [friction_angle]
    })

    try:
        # Predict
        prediction = model.predict(input_data)
        fos = prediction[0]

        st.header(f"Factor of Safety (FoS): {fos:.3f}")

        # Interpretation Visuals
        if fos < 1.0:
            st.error("❌ FAILURE (FoS < 1.0)")
            st.write("The driving forces exceed resisting forces. Redesign required.")
        elif fos < 1.3:
            st.warning("⚠️ MARGINAL / UNSAFE (1.0 ≤ FoS < 1.3)")
            st.write("Wall is temporarily stable but does not meet standard safety margins.")
        elif fos < 1.5:
            st.success("✅ STABLE (1.3 ≤ FoS < 1.5)")
            st.write("Acceptable for temporary excavation structures.")
        else:
            st.success("🌟 HIGHLY STABLE (FoS ≥ 1.5)")
            st.write("Acceptable for permanent structural support.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure the model file uploaded is the newly trained one and matches these exactly 8 features.")