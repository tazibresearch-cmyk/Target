import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="High-Performance Biopolymer Metamodel Studio", layout="wide")

# ==========================================
# 1. COMPREHENSIVE MASTER DATA MATRIX
# ==========================================
# [PBAT%, PBS%, Granite%, Plast_Type, GEL%, RF%, OBE%, Is_Xray, Dose, Thickness, TS_m, TS_s, Mod_m, Mod_s, Elong_m, Elong_s, OTR_m, OTR_s]
# Plast_Type: 0=Gly, 1=TBC, 2=CARD, 3=Commercial Control
master_source_matrix = np.array([
    # --- Glycerol Blends (Unirradiated Base Controls) ---
    [100, 0,  0,  3, 0, 0,    0, 0, 0,    0.198, 31.86, 2.31,  3.84,  0.45,  644.66, 9.29,  12.72,  0.15],  # Commercial Bergeron
    [100, 0,  5,  0, 0, 0,    0, 0, 0,    0.130, 13.86, 1.80,  1.34,  0.27,  542.33, 20.4,  426.0,   41.0],
    [100, 0,  10, 0, 0, 0,    0, 0, 0,    0.133, 14.80, 5.20,  1.63,  0.46,  513.0,  65.2,  365.0,   14.0],
    [100, 0,  15, 0, 0, 0,    0, 0, 0,    0.135, 12.90, 1.70,  0.96,  0.07,  542.31, 58.1,  140.0,   12.6],
    [95,  5,  5,  0, 0, 0,    0, 0, 0,    0.128, 10.40, 0.51,  0.69,  0.09,  470.6,  53.5,  369.25,  20.1],
    [95,  5,  10, 0, 0, 0,    0, 0, 0,    0.125, 10.04, 1.93,  1.40,  0.61,  220.0,  28.0,  567.5,   31.8],
    [95,  5,  15, 0, 0, 0,    0, 0, 0,    0.118, 10.33, 2.55,  0.53,  0.02,  367.6,  18.9,  268.5,   2.1],
    [90,  10, 5,  0, 0, 0,    0, 0, 0,    0.131, 7.04,  0.40,  2.13,  0.33,  171.6,  26.08, 1296.5,  27.5],
    [90,  10, 10, 0, 0, 0,    0, 0, 0,    0.135, 9.36,  1.10,  1.06,  0.38,  321.3,  48.2,  292.5,   3.5],
    [90,  10, 15, 0, 0, 0,    0, 0, 0,    0.133, 9.20,  1.05,  0.835, 0.18,  334.3,  30.2,  183.83,  16.5],

    # --- Tributyl Citrate (TBC) Blends ---
    [100, 0,  5,  1, 0, 0,    0, 0, 0,    0.095, 10.56, 0.20,  0.899, 0.19,  345.0,  36.3,  1224.0,  19.7],
    [100, 0,  10, 1, 0, 0,    0, 0, 0,    0.105, 7.71,  0.96,  0.704, 0.01,  556.3,  28.7,  332.5,   31.8],
    [100, 0,  15, 1, 0, 0,    0, 0, 0,    0.102, 5.41,  0.25,  0.74,  0.03,  305.3,  5.6,   333.3,   31.0],
    [95,  5,  5,  1, 0, 0,    0, 0, 0,    0.091, 7.29,  1.70,  0.63,  0.01,  412.6,  33.2,  1500.5,  71.4],
    [95,  5,  10, 1, 0, 0,    0, 0, 0,    0.088, 5.78,  0.62,  0.64,  0.01,  322.3,  33.3,  510.0,   77.8],
    [95,  5,  15, 1, 0, 0,    0, 0, 0,    0.105, 7.83,  1.30,  0.50,  0.03,  514.6,  28.9,  109.5,   11.5],
    [90,  10, 5,  1, 0, 0,    0, 0, 0,    0.101, 6.64,  1.10,  2.99,  0.51,  220.3,  15.3,  1428.0,  39.59],
    [90,  10, 10, 1, 0, 0,    0, 0, 0,    0.092, 12.93, 0.60,  0.73,  0.06,  585.33, 18.5,  1195.5,  9.2],
    [90,  10, 15, 1, 0, 0,    0, 0, 0,    0.121, 8.60,  0.10,  0.71,  0.03,  377.3,  11.1,  1555.0,  77.7],

    # --- Coated Bioactive Films with Active Crosslinking ---
    [95,  5,  15, 0, 0, 0,    0, 0, 0,    0.250, 12.83, 0.95,  1.70,  0.08,  272.33, 52.77, 288.56,  9.1],
    [95,  5,  15, 0, 8, 0,    0, 0, 0,    0.230, 12.37, 0.32,  1.21,  0.81,  246.33, 41.50, 15.1,    1.5],
    [95,  5,  15, 0, 8, 1.25, 0, 0, 1.92,  0.272, 10.57, 0.15,  14.86, 3.90,  170.0,  49.49, 286.25,  8.8],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 4.32,  0.240, 12.77, 0.15,  2.247, 1.29,  220.67, 16.26, 126.98,  7.09],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 6.24,  0.278, 10.67, 1.14,  33.16, 9.60,  84.47,  15.84, 43.55,   2.04],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 8.16,  0.241, 12.57, 0.65,  1.96,  0.60,  289.33, 49.89, 12.09,   1.2],
    [95,  5,  15, 0, 4, 1.25, 0, 0, 10.56, 0.237, 11.27, 0.61,  1.51,  0.27,  190.67, 29.69, 14.97,   1.3],
    [95,  5,  15, 0, 4, 1.25, 5, 0, 8.16,  0.301, 6.58,  0.43,  41.53, 7.11,  51.63,  8.06,  563.5,   50.2],
    [95,  5,  15, 0, 4, 0,    0, 1, 0,    0.250, 12.83, 0.95,  1.70,  0.08,  272.33, 52.77, 288.56,  9.1],
    [95,  5,  15, 0, 4, 0,    0, 1, 0,    0.230, 12.37, 0.32,  1.21,  0.81,  246.33, 41.50, 15.1,    1.5],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 0.5,   0.233, 10.71, 1.04,  1.63,  0.47,  238.33, 47.42, 7.12,    1.71],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 1.0,   0.265, 10.14, 0.84,  5.56,  0.54,  146.67, 20.11, 16.62,   1.66],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 2.5,   0.225, 9.987, 0.533, 7.783, 6.591, 146.0,  47.466, 10.25,   1.33],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 5.0,   0.251, 8.36,  1.40,  3.607, 3.11,  127.0,  16.37, 15.14,   1.91],
    [95,  5,  15, 0, 4, 1.25, 0, 1, 10.0,  0.266, 10.10, 0.878, 3.967, 3.70,  186.67, 37.61, 149.07,  8.38]
])

# ==========================================
# 2. MONTE CARLO ENGINE (1,200 SAMPLES PER GROUP)
# ==========================================
@st.cache_data
def generate_monte_carlo_dataset(replicates=1200):
    np.random.seed(42)
    rows = []
    
    for r in master_source_matrix:
        features = r[:10]
        ts_m, ts_s, mod_m, mod_s, el_m, el_s, OTR_m, OTR_s = r[10:]
        
        for _ in range(replicates):
            ts = max(0.1, np.random.normal(ts_m, ts_s))
            mod = max(0.01, np.random.normal(mod_m, mod_s))
            el = max(1.0, np.random.normal(el_m, el_s))
            otr = max(0.1, np.random.normal(OTR_m, OTR_s))
            
            rows.append(list(features) + [ts, mod, el, otr])
            
    columns = ['PBAT', 'PBS', 'Granite', 'Plasticizer_Type', 'GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness',
               'Tensile_Strength', 'Modulus', 'Elongation', 'OTR']
    return pd.DataFrame(rows, columns=columns)

df_synthetic = generate_monte_carlo_dataset()

# ==========================================
# 3. ADVANCED GBM METAMODEL ENGINE (UPGRADED)
# ==========================================
@st.cache_resource
def train_high_effectiveness_models(df):
    features = ['PBAT', 'PBS', 'Granite', 'Plasticizer_Type', 'GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness']
    targets = ['Tensile_Strength', 'Modulus', 'Elongation', 'OTR']
    
    X = df[features]
    
    # Feature Scaling ensures minor structural variations (like thickness) are heavily weighted
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    trained_models = {}
    for target in targets:
        # Upgraded to Gradient Boosting Regressors for extreme prediction fidelity
        gbm = GradientBoostingRegressor(
            n_estimators=60, 
            learning_rate=0.1, 
            max_depth=6, 
            min_samples_split=4,
            random_state=42
        )
        gbm.fit(X_scaled, df[target])
        trained_models[target] = gbm
        
    return trained_models, scaler

models, feature_scaler = train_high_effectiveness_models(df_synthetic)

# ==========================================
# 4. STREAMLIT APP UI INTERFACE
# ==========================================
st.title("🚀 Advanced Biopolymer Film Formulation Studio")
st.markdown("---")

# Sidebar configurations
st.sidebar.header("🧱 Polymer Base & Filler Matrix")
pbat = st.sidebar.slider("PBAT Blend Fraction (%)", 90, 100, 95, 5)
pbs = 100 - pbat
st.sidebar.text(f"Calculated PBS Balance: {pbs}%")

granite = st.sidebar.slider("Yellow Granite Filler Content (%)", 5, 15, 15, 5)

plasticizer_label = st.sidebar.selectbox("Plasticizer Selection", ["Glycerol (Gly)", "Tributyl Citrate (TBC)", "Cardanol (CARD)", "Commercial Reference"])
plasticizer_map = {"Glycerol (Gly)": 0, "Tributyl Citrate (TBC)": 1, "Cardanol (CARD)": 2, "Commercial Reference": 3}
plast_type = plasticizer_map[plasticizer_label]

st.sidebar.header("🛠️ Coating & Radiation Controls")
gel = st.sidebar.slider("Gelatin Coating (GEL %)", 0.0, 8.0, 4.0, 4.0)
rf = st.sidebar.slider("Riboflavin (RF %)", 0.0, 1.25, 1.25, 1.25)
obe = st.sidebar.slider("Onion-Broccoli Extract (OBE %)", 0.0, 5.0, 0.0, 5.0)
thickness = st.sidebar.slider("Film Thickness Layer (mm)", 0.08, 0.35, 0.15, 0.01)

rad_selection = st.sidebar.selectbox("Radiation Target", ["UV-C (J/cm²)", "X-Ray (kGy)"])
is_xray = 1.0 if "X-Ray" in rad_selection else 0.0
dose = st.sidebar.slider("Active Crosslinking Dose", 0.0, 11.0, 4.0, 0.1)

# Format the current user query through the scale-vector engine
query_raw = np.array([[pbat, pbs, granite, plast_type, gel, rf, obe, is_xray, dose, thickness]])
query_scaled = feature_scaler.transform(query_raw)

predictions = {}
for target, model in models.items():
    predictions[target] = float(model.predict(query_scaled)[0])

tab1, tab2, tab3 = st.tabs(["📊 Property Dashboard", "🌌 3D Material Interaction Mesh", "🧀 Cheese Packaging Feasibility"])

with tab1:
    st.subheader("🔮 Metamodel Performance Predictions")
    col1, col2 = st.columns(2)
    with col1:
        st.info("⚓ Mechanical Properties")
        st.metric("Tensile Strength", f"{predictions['Tensile_Strength']:.2f} MPa")
        st.metric("Elastic Modulus", f"{predictions['Modulus']:.2f} MPa")
        st.metric("Elongation at Break", f"{predictions['Elongation']:.1f} %")
    with col2:
        st.warning("🛡️ Barrier Capabilities")
        st.metric("Oxygen Transmission Rate (OTR)", f"{predictions['OTR']:.2f} cc/m².day")

with tab2:
    st.subheader("🌌 Multi-Variable 3D Optimization Surface Mesh")
    plot_target = st.selectbox("Select Optimization Metric", ['Elongation', 'OTR', 'Tensile_Strength'])
    
    dose_range = np.linspace(0, 11, 20)
    granite_range = np.linspace(5, 15, 20)
    D, G = np.meshgrid(dose_range, granite_range)
    
    mesh_rows = []
    for g_val, d_val in zip(G.ravel(), D.ravel()):
        mesh_rows.append([pbat, pbs, g_val, plast_type, gel, rf, obe, is_xray, d_val, thickness])
        
    mesh_scaled = feature_scaler.transform(np.array(mesh_rows))
    mesh_preds = models[plot_target].predict(mesh_scaled).reshape(20, 20)
    
    fig = go.Figure(data=[go.Surface(z=mesh_preds, x=dose_range, y=granite_range, colorscale='Viridis')])
    fig.update_layout(
        scene=dict(xaxis_title="Radiation Dose", yaxis_title="Yellow Granite Filler (%)", zaxis_title=plot_target),
        width=800, height=600
    )
