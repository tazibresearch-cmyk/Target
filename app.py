import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Set up browser window layout
st.set_page_config(page_title="Biopolymer Metamodel Studio", layout="wide")

# ==========================================
# 1. SCIENTIFICALLY ALIGNED SOURCE MATRIX
# ==========================================
# Encoding arrays:
# Features: [PBAT%, PBS%, YG%, Plasticizer, GEL%, RF%, OBE%, Is_Xray, Dose, Thickness]
# Plasticizer mapping: 0 = Glycerol (Gly), 1 = Tributyl Citrate (TBC), 3 = Commercial Control
# Targets:   [TS_m, TS_s, Mod_m, Mod_s, Elong_m, Elong_s, OTR_m, OTR_s, Tm1, Tg, Tc, Tmax]
source_matrix = np.array([
    # --- PHASE 1: STRUCTURAL BASE BLENDS (Uncoated, Unirradiated Controls) ---
    [100, 0,  0,  3, 0, 0,    0, 0, 0,    0.198,  31.86, 2.31,  3.84,  0.45,  644.66, 9.29,  12.72,  0.15,  44.9,  -31.86, 68.7,  418.72], # Bergeron
    [100, 0,  5,  0, 0, 0,    0, 0, 0,    0.130,  13.86, 1.80,  1.34,  0.27,  542.33, 20.4,  426.0,   41.0,  44.9,  -31.86, 68.7,  418.72],
    [100, 0,  10, 0, 0, 0,    0, 0, 0,    0.133,  14.80, 5.20,  1.63,  0.46,  513.0,  65.2,  365.0,   14.0,  44.9,  -31.86, 68.7,  418.72],
    [100, 0,  15, 0, 0, 0,    0, 0, 0,    0.135,  12.90, 1.70,  0.96,  0.07,  542.31, 58.1,  140.0,   12.6,  44.9,  -31.86, 68.7,  418.72],
    [95,  5,  5,  0, 0, 0,    0, 0, 0,    0.128,  10.40, 0.51,  0.69,  0.09,  470.6,  53.5,  369.25,  20.1,  44.9,  -31.86, 68.7,  418.72],
    [95,  5,  10, 0, 0, 0,    0, 0, 0,    0.125,  10.04, 1.93,  1.40,  0.61,  220.0,  28.0,  567.5,   31.8,  44.9,  -31.86, 68.7,  418.72],
    [95,  5,  15, 0, 0, 0,    0, 0, 0,    0.118,  10.33, 2.55,  0.53,  0.02,  367.6,  18.9,  268.5,   2.1,   44.9,  -31.86, 68.7,  418.72],
    [90,  10, 5,  0, 0, 0,    0, 0, 0,    0.131,  7.04,  0.40,  2.13,  0.33,  171.6,  26.08, 1296.5,  27.5,  44.9,  -31.86, 68.7,  418.72],
    [90,  10, 10, 0, 0, 0,    0, 0, 0,    0.135,  9.36,  1.10,  1.06,  0.38,  321.3,  48.2,  292.5,   3.5,   44.9,  -31.86, 68.7,  418.72],
    [90,  10, 15, 0, 0, 0,    0, 0, 0,    0.133,  9.20,  1.05,  0.835, 0.18,  334.3,  30.2,  183.83,  16.5,  44.9,  -31.86, 68.7,  418.72],
    [100, 0,  5,  1, 0, 0,    0, 0, 0,    0.095,  10.56, 0.20,  0.899, 0.19,  345.0,  36.3,  1224.0,  19.7,  44.9,  -31.86, 68.7,  418.72],
    [100, 0,  10, 1, 0, 0,    0, 0, 0,    0.105,  7.71,  0.96,  0.704, 0.01,  556.3,  28.7,  332.5,   31.8,  44.9,  -31.86, 68.7,  418.72],
    [100, 0,  15, 1, 0, 0,    0, 0, 0,    0.102,  5.41,  0.25,  0.74,  0.03,  305.3,  5.6,   333.3,   31.0,  44.9,  -31.86, 68.7,  418.72],
    [95,  5,  5,  1, 0, 0,    0, 0, 0,    0.091,  7.29,  1.70,  0.63,  0.01,  412.6,  33.2,  1500.5,  71.4,  44.9,  -31.86, 68.7,  418.72],
    [95,  5,  10, 1, 0, 0,    0, 0, 0,    0.088,  5.78,  0.62,  0.64,  0.01,  322.3,  33.3,  510.0,   77.8,  44.9,  -31.86, 68.7,  418.72],
    [95,  5,  15, 1, 0, 0,    0, 0, 0,    0.105,  7.83,  1.30,  0.50,  0.03,  514.6,  28.9,  109.5,   11.5,  44.9,  -31.86, 68.7,  418.72],
    [90,  10, 5,  1, 0, 0,    0, 0, 0,    0.101,  6.64,  1.10,  2.99,  0.51,  220.3,  15.3,  1428.0,  39.59, 44.9,  -31.86, 68.7,  418.72],
    [90,  10, 10, 1, 0, 0,    0, 0, 0,    0.092,  12.93, 0.60,  0.73,  0.06,  585.33, 18.5,  1195.5,  9.2,   44.9,  -31.86, 68.7,  418.72],
    [90,  10, 15, 1, 0, 0,    0, 0, 0,    0.121,  8.60,  0.10,  0.71,  0.03,  377.3,  11.1,  1555.0,  77.7,  44.9,  -31.86, 68.7,  418.72],

    # --- PHASE 2A: UV-C IRRADIATED FUNCTIONAL BIO-FILMS (Locked Structural Matrix PBAT95/PBS5/YG15) ---
    [95,  5,  15, 0, 8, 0,    0, 0, 0,    0.230,  12.37, 0.32,  1.21,  0.81,  246.33, 41.50, 15.1,    1.5,   44.75, -32.48, 70.7,  417.97],
    [95,  5,  15, 0, 8, 1.25, 0, 0, 1.92,  0.272,  10.57, 0.15,  14.86, 3.90,  170.0,  49.49, 286.25,  8.8,   44.9,  -32.18, 69.8,  418.72],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 4.32,  0.240,  12.77, 0.15,  2.247, 1.29,  220.67, 16.26, 126.98,  7.09,  44.7,  -33.43, 71.4,  421.25],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 6.24,  0.278,  10.67, 1.14,  33.16, 9.60,  84.47,  15.84, 43.55,   2.04,  47.8,  -32.73, 72.5,  415.41],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 8.16,  0.241,  12.57, 0.65,  1.96,  0.60,  289.33, 49.89, 12.09,   1.2,   46.71, -32.76, 70.26, 418.25],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 10.56, 0.237,  11.27, 0.61,  1.51,  0.27,  190.67, 29.69, 14.97,   1.3,   45.2,  -33.78, 69.2,  418.06],
    [95,  5,  15, 0, 4, 1.25, 5, 0, 8.16,  0.301,  6.58,  0.43,  41.53, 7.11,  51.63,  8.06,  563.5,   50.2,  45.7,  -34.8,  71.9,  417.85],

    # --- PHASE 2B: X-RAY IRRADIATED FUNCTIONAL BIO-FILMS (Locked Structural Matrix PBAT95/PBS5/YG15) ---
    [95,  5,  15, 0, 4, 1.25, 0, 1, 0.5,   0.233,  10.71, 1.04,  1.63,  0.47,  238.33, 47.42, 7.12,    1.71,  43.5,  -34.2,  69.5,  420.22],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 1.0,   0.265,  10.14, 0.84,  5.56,  0.54,  146.67, 20.11, 16.62,   1.66,  45.3,  -32.75, 73.8,  418.37],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 2.5,   0.225,  9.987, 0.533, 7.783, 6.591, 146.0,  47.466, 10.25,   1.33,  50.6,  -33.38, 73.0,  419.79],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 5.0,   0.251,  8.36,  1.40,  3.607, 3.11,  127.0,  16.37, 15.14,   1.91,  49.7,  -34.01, 69.6,  418.06],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 10.0,  0.266,  10.10, 0.878, 3.967, 3.70,  186.67, 37.61, 149.07,  8.38,  50.6,  -33.5,  71.8,  420.54]
])

# ==========================================
# 2. DATA GENERATOR (1,200 REPLICATES / CONDITION)
# ==========================================
@st.cache_data
def run_monte_carlo_simulation(replicates=1200):
    np.random.seed(42)
    synthetic_rows = []
    
    for row in source_matrix:
        features = row[:10]
        ts_m, ts_s, mod_m, mod_s, el_m, el_s, OTR_m, OTR_s = row[10:18]
        tm1, tg, tc, tmax = row[18:]
        
        for _ in range(replicates):
            ts = max(0.1, np.random.normal(ts_m, ts_s))
            mod = max(0.01, np.random.normal(mod_m, mod_s))
            el = max(1.0, np.random.normal(el_m, el_s))
            otr = max(0.1, np.random.normal(OTR_m, OTR_s))
            
            s_tm1 = np.random.normal(tm1, 0.4)
            s_tg  = np.random.normal(tg, 0.3)
            s_tc  = np.random.normal(tc, 0.4)
            s_tmax = np.random.normal(tmax, 0.8)
            
            synthetic_rows.append(list(features) + [ts, mod, el, otr, s_tm1, s_tg, s_tc, s_tmax])
            
    columns = ['PBAT', 'PBS', 'Granite', 'Plasticizer_Type', 'GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness',
               'Tensile_Strength', 'Modulus', 'Elongation', 'OTR', 'Tm1', 'Tg', 'Tc', 'Tmax']
    return pd.DataFrame(synthetic_rows, columns=columns)

df_synthetic = run_monte_carlo_simulation()

# ==========================================
# 3. ROBUST PACKAGING MACHINE LEARNING ENGINE
# ==========================================
@st.cache_resource
def train_gradient_boosting_metamodels(df):
    features = ['PBAT', 'PBS', 'Granite', 'Plasticizer_Type', 'GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness']
    targets = ['Tensile_Strength', 'Modulus', 'Elongation', 'OTR', 'Tm1', 'Tg', 'Tc', 'Tmax']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    trained_models = {}
    for target in targets:
        gbm = GradientBoostingRegressor(n_estimators=65, learning_rate=0.08, max_depth=6, random_state=42)
        gbm.fit(X_scaled, df[target])
        trained_models[target] = gbm
        
    return trained_models, scaler

models, feature_scaler = train_gradient_boosting_metamodels(df_synthetic)

# ==========================================
# 4. APP DASHBOARD LAYOUT & DESIGN
# ==========================================
st.title("🔬 Advanced Biopolymer Metamodel Calculator Studio")
st.markdown("This machine learning framework tracks structural variations, crosslinking treatments, and bioactive film dynamics.")
st.markdown("---")

# User Configuration Side Panel
st.sidebar.header("🧱 Formulation Matrix Settings")
pbat = st.sidebar.slider("PBAT Base Percentage (%)", 90, 100, 95, 5)
pbs = 100 - pbat
st.sidebar.text(f"Calculated PBS Balance fraction: {pbs}%")

granite = st.sidebar.slider("Yellow Granite (YG) Content (%)", 5, 15, 15, 5)

plasticizer = st.sidebar.selectbox("Plasticizer Matrix Element", ["Glycerol (Gly)", "Tributyl Citrate (TBC)", "Commercial Control File"])
plast_map = {"Glycerol (Gly)": 0, "Tributyl Citrate (TBC)": 1, "Commercial Control File": 3}
plast_type = plast_map[plasticizer]

st.sidebar.header("⚡ Coating & Crosslinking Energy Parameters")
gel = st.sidebar.slider("Gelatin Top Coat Content (GEL %)", 0.0, 8.0, 4.0, 4.0)
rf = st.sidebar.slider("Riboflavin Crosslinker Agent (RF %)", 0.0, 1.25, 1.25, 1.25)
obe = st.sidebar.slider("Onion-Broccoli Extract (OBE %)", 0.0, 5.0, 0.0, 5.0)
thickness = st.sidebar.slider("Film Profile Layer Thickness (mm)", 0.08, 0.35, 0.24, 0.01)

rad_selection = st.sidebar.selectbox("Irradiation Configuration Spectrum", ["UV-C (J/cm²)", "X-Ray (kGy)"])
is_xray = 1.0 if "X-Ray" in rad_selection else 0.0
dose = st.sidebar.slider("Active Energy Exposure Level", 0.0, 11.0, 4.32, 0.1)

# Format the inputs for prediction
query_raw = np.array([[pbat, pbs, granite, plast_type, gel, rf, obe, is_xray, dose, thickness]])
query_scaled = feature_scaler.transform(query_raw)

# FIXED: Explicitly indexing using [0] handles array-to-float conversions cleanly
predictions = {}
for target, model in models.items():
    predictions[target] = float(model.predict(query_scaled)[0])

tab1, tab2, tab3 = st.tabs(["📊 Performance Parameter Engine", "🌌 3D Energy Interaction Mesh", "🧀 Food Technology Assessment"])

# --- TAB 1: LIVE INTERACTIVE PREDICTION ENGINE ---
with tab1:
    st.subheader("🔮 Metamodel Performance Diagnostics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("⚓ Mechanical Properties")
        st.metric("Tensile Strength (TS)", f"{predictions['Tensile_Strength']:.2f} MPa")
