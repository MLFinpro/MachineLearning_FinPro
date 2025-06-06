
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# --- Configuration and Constants ---
B_BASE = 500
T0_BASELINE_TEMP = 25
H0_BASELINE_HUMID = 50
ALPHA_TEMP_FACTOR = 0.02
BETA_HUMID_FACTOR = 0.01

PREFIX_TO_PHYSICAL_TYPE_MAP = {
    'PLFY': 'Ceiling_Cassette',
    'PKFY': 'Wall_Mounted',
    'PEFY': 'Ceiling_Concealed_Ducted',
    'PCFY': 'Ceiling_Suspended',
    'PFFY': 'Floor_Standing'
    # Tambahkan prefix lainnya di sini jika perlu
}

@st.cache_data
def load_and_preprocess_ac_data(csv_path='datasetAc(datasetAc).csv'):
    try:
        ac_df_original = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: AC dataset '{csv_path}' not found.")
        return None

    ac_df = ac_df_original.copy()
    if ac_df.columns[-1].startswith('Unnamed'):
        ac_df = ac_df.iloc[:, :-1]
    ac_df = ac_df[['AC name', 'cooling capacity BTU/h', 'power input kW']]
    ac_df.columns = ['AC_Name', 'BTU_Capacity', 'Power_Input_kW']
    ac_df = ac_df.dropna()
    ac_df['BTU_Capacity'] = pd.to_numeric(ac_df['BTU_Capacity'], errors='coerce')
    ac_df['Power_Input_kW'] = pd.to_numeric(ac_df['Power_Input_kW'], errors='coerce')
    ac_df = ac_df.dropna(subset=['BTU_Capacity', 'Power_Input_kW'])
    ac_df = ac_df[ac_df['BTU_Capacity'] > 0]
    ac_df = ac_df[ac_df['Power_Input_kW'] > 0]
    ac_df = ac_df.reset_index(drop=True)

    def get_ac_prefix(ac_name):
        if isinstance(ac_name, str): return ac_name.split('-')[0].upper()
        return "UNKNOWN_PREFIX"
    ac_df['AC_Prefix'] = ac_df['AC_Name'].apply(get_ac_prefix)
    ac_df['Physical_Type'] = ac_df['AC_Prefix'].map(PREFIX_TO_PHYSICAL_TYPE_MAP).fillna('Other_Unknown')

    def get_cap_label(btu):
        if btu < 9000: return 'Small'
        elif btu < 18000: return 'Medium'
        elif btu < 36000: return 'Large'
        else: return 'Very_Large'
    ac_df['Capacity_Label'] = ac_df['BTU_Capacity'].apply(get_cap_label)
    ac_df['AC_Type_Category'] = ac_df['Capacity_Label'] + "_" + ac_df['Physical_Type']

    return ac_df

def calculate_required_btu(temp, humid, volume):
    return max(0, B_BASE * volume * (1 + ALPHA_TEMP_FACTOR * (temp - T0_BASELINE_TEMP)) * (1 + BETA_HUMID_FACTOR * (humid - H0_BASELINE_HUMID)))

def get_capacity_label_for_btu_value(btu_value):
    if btu_value < 9000: return 'Small'
    elif btu_value < 18000: return 'Medium'
    elif btu_value < 36000: return 'Large'
    else: return 'Very_Large'

@st.cache_resource
def load_prediction_assets():
    try:
        model = joblib.load('ac_type_classifier_model_v4.pkl')
        scaler = joblib.load('feature_scaler_v4.pkl')
        label_encoder = joblib.load('ac_type_label_encoder_v4.pkl')
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("Error: Model assets (v4 .pkl files) not found.")
        return None, None, None

def generate_recommendation_details(temp, humid, area, height,
                                    trained_model, feature_scaler, category_encoder, ac_data_full,
                                    preferred_physical_type=None):
    messages = []
    recommended_df = pd.DataFrame()
    volume = area * height
    btu_required = calculate_required_btu(temp, humid, volume)

    messages.append(f"**Input Conditions:** Temp={temp}°C, Humid={humid}%, Area={area}m², Height={height}m, Volume={volume:.2f}m³")
    messages.append(f"**Calculated Required BTU:** {btu_required:.2f}")

    input_features_ml = np.array([[temp, humid, area, height, volume, btu_required]])
    input_features_ml_scaled = feature_scaler.transform(input_features_ml)
    ml_predicted_category_encoded = trained_model.predict(input_features_ml_scaled)
    ml_predicted_category = category_encoder.inverse_transform(ml_predicted_category_encoded)[0]
    messages.append(f"**ML's Initial Suggested AC Type Category:** `{ml_predicted_category}`")

    final_search_category = ml_predicted_category
    search_basis_message = f"Using ML's suggestion: `{ml_predicted_category}`"

    if preferred_physical_type and preferred_physical_type != "Any (Let ML Decide)":
        if preferred_physical_type not in ac_data_full['Physical_Type'].unique():
            messages.append(f":warning: Preferred physical type '{preferred_physical_type}' is not recognized. Using ML suggestion.")
        else:
            capacity_label = get_capacity_label_for_btu_value(btu_required)
            final_search_category = f"{capacity_label}_{preferred_physical_type}"
            search_basis_message = f"Using user preference: `{final_search_category}`"

    messages.append(f"**Searching for ACs based on:** {search_basis_message}")

    recommended_acs_df = ac_data_full[
        (ac_data_full['AC_Type_Category'] == final_search_category) &
        (ac_data_full['BTU_Capacity'] >= btu_required) &
        (ac_data_full['BTU_Capacity'] <= btu_required * 1.30)
    ]

    if recommended_acs_df.empty and preferred_physical_type and preferred_physical_type != "Any (Let ML Decide)":
        messages.append("No exact match. Trying all with same physical type...")
        recommended_acs_df = ac_data_full[
            (ac_data_full['Physical_Type'] == preferred_physical_type) &
            (ac_data_full['BTU_Capacity'] >= btu_required) &
            (ac_data_full['BTU_Capacity'] <= btu_required * 1.35)
        ]

    if recommended_acs_df.empty:
        messages.append("Still no match. Broadening search...")
        recommended_acs_df = ac_data_full[
            (ac_data_full['BTU_Capacity'] >= btu_required) &
            (ac_data_full['BTU_Capacity'] <= btu_required * 1.40)
        ]

    if not recommended_acs_df.empty:
        recommended_acs_df = recommended_acs_df.sort_values(by=['Power_Input_kW', 'BTU_Capacity'])
        recommended_df = recommended_acs_df[['AC_Name', 'BTU_Capacity', 'Power_Input_kW', 'AC_Type_Category', 'Physical_Type']].head()
    else:
        messages.append(":x: No suitable AC models found.")

    return messages, recommended_df

# --- Streamlit App UI ---
st.set_page_config(page_title="Advanced AC Recommender", layout="wide")
st.title("🌬️ Advanced Air Conditioner Recommender")
st.write("Enter room details and preferences to get tailored AC recommendations using our v4 Hybrid Model.")

ac_df = load_and_preprocess_ac_data()
model, scaler, label_encoder = load_prediction_assets()

if ac_df is not None and model is not None:
    st.sidebar.header("📊 Room & Preference Inputs:")
    available_physical_types = ["Any (Let ML Decide)"] + sorted(ac_df['Physical_Type'].unique())

    input_temp = st.sidebar.number_input("Room Temperature (°C)", min_value=10.0, max_value=40.0, value=25.0)
    input_humid = st.sidebar.number_input("Room Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    input_area = st.sidebar.number_input("Room Area (m²)", min_value=5.0, max_value=200.0, value=20.0)
    input_height = st.sidebar.number_input("Room Height (m)", min_value=2.0, max_value=5.0, value=2.8)
    input_preferred_type = st.sidebar.selectbox("Preferred AC Physical Type", options=available_physical_types)

    if st.sidebar.button("🔍 Recommend AC"):
        st.markdown("---")
        st.subheader("📋 Recommendation Results")
        messages, recommended_df = generate_recommendation_details(
            input_temp, input_humid, input_area, input_height,
            model, scaler, label_encoder, ac_df, input_preferred_type
        )

        for msg in messages:
            st.markdown(msg, unsafe_allow_html=True)

        if not recommended_df.empty:
            st.dataframe(recommended_df, use_container_width=True)



