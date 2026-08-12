import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Set up page configurations
st.set_page_config(page_title="Biopolymer Film Metamodel Studio", layout="wide")

# ==========================================
# 1. COMPREHENSIVE EXPERIMENTAL SOURCE MATRIX
# ==========================================
source_matrix = np.array([
    # --- UV-C Groups (Dose in J/cm², Is_Xray = 0) ---
    [0, 0,    0, 0, 0,     0.25,  12.83, 0.95,  1.70,  0.08,  272.33, 52.77, 2.012, 0.35,  288.56, 9.1,  44.9,  113.05, 134.3,  -31.86, 68.7,  418.72],
    [8, 0,    0, 0, 0,     0.23,  12.37, 0.32,  1.21,  0.81,  246.33, 41.50, 2.78,  0.006, 15.1,   1.5,  44.75, 111.65, 132.48, -32.48, 70.7,  417.97],
    [8, 1.25, 0, 0, 1.92,  0.272, 10.57, 0.15,  14.86, 3.90,  170.00, 49.49, 2.05,  0.10,  286.25, 8.8,  44.9,  112.1,  131.73, -32.18, 69.8,  418.72],
    [4, 1.25, 0, 0, 4.32,  0.24,  12.77, 0.15,  2.247, 1.29,  220.67, 16.26, 1.63,  0.54,  126.98, 7.09, 44.7,  112.7,  133.2,  -33.43, 71.4,  421.25],
    [4, 1.25, 0, 0, 6.24,  0.278, 10.67, 1.14,  33.16, 9.60,  84.47,  15.84, 1.77,  0.70,  43.55,  2.04, 47.8,  112.4,  133.03, -32.73, 72.5,  415.41],
    [4, 1.25, 0, 0, 8.16,  0.241, 12.57, 0.65,  1.96,  0.60,  289.33, 49.89, 0.73,  0.10,  12.09,  1.2,  46.71, 112.56, 133.2,  -32.76, 70.26, 418.25],
    [4, 1.25, 0, 0, 10.56, 0.237, 11.27, 0.61,  1.51,  0.27,  190.67, 29.69, 2.34,  0.10,  14.97,  1.3,  45.2,  113.0,  133.03, -33.78, 69.2,  418.06],
    [4, 1.25, 5, 0, 8.16,  0.301, 6.58,  0.43,  41.53, 7.114, 51.63,  8.06,  3.34,  0.10,  563.5,  50.2, 45.7,  112.3,  132.2,  -34.8,  71.9,  417.85],
    
    # --- X-Ray Groups (Dose in kGy, Is_Xray = 1) ---
    [4, 0,    0, 1, 0,     0.25,  12.83, 0.95,  1.70,  0.08,  272.33, 52.77, 2.012, 0.35,  288.56, 9.1,  44.9,  113.05, 134.3,  -31.86, 68.7,  418.72],
    [4, 0,    0, 1, 0,     0.23,  12.37, 0.32,  1.21,  0.81,  246.33, 41.50, 2.78,  0.006, 15.1,   1.5,  44.9,  113.05, 134.3,  -31.86, 68.7,  418.72],
    [4, 1.25, 0, 1, 0.5,   0.233, 10.71, 1.04,  1.63,  0.47,  238.33, 47.42, 0.82,  0.04,  7.12,   1.71, 43.5,  112.1,  132.3,  -34.2,  69.5,  420.22],
    [4, 1.25, 0, 1, 1.0,   0.265, 10.14, 0.84,  5.56,  0.54,  146.67, 20.11, 2.56,  0.66,  16.62,  1.66, 45.3,  110.7,  130.78, -32.75, 73.8,  418.37],
    [4, 1.25, 0, 1, 2.5,   0.225, 9.987, 0.533, 7.783, 6.591, 146.00, 47.466,3.14,  0.48,  10.25,  1.33, 50.6,  111.8,  133.6,  -33.38, 73.0,  419.79],
    [4, 1.25, 0, 1, 5.0,   0.251, 8.36,  1.40,  3.607, 3.11,  127.00, 16.37, 2.01,  0.02,  15.14,  1.91, 49.7,  112.7,  132.06, -34.01, 69.6,  418.06],
    [4, 1.25, 0, 1, 10.0,  0.266, 10.10, 0.878, 3.967, 3.70,  186.67, 37.61, 1.58,  0.32,  149.07, 8.38, 50.6,  113.1,  132.16, -33.5,  71.8,  420.54]
])

# ==========================================
# 2. CACHED DATASETS GENERATOR (>1000 REPLICATES)
# ==========================================
@st.cache_data
def generate_monte_carlo_dataset(replicates=1100):
    np.random.seed(42)
    rows = []
    
    for r in source_matrix:
        feat = r[:6]
        ts_m, ts_s, mod_m, mod_s, el_m, el_s, wvp_m, wvp_s, otr_m, otr_s = r[6:16]
        tm1, tm2, tm3, tg, tc, tmax = r[16:]
        
        for _ in range(replicates):
            ts = max(0.5, np.random.normal(ts_m, ts_s))
            mod = max(0.1, np.random.normal(mod_m, mod_s))
            el = max(5.0, np.random.normal(el_m, el_s))
            wvp = max(0.05, np.random.normal(wvp_m, wvp_s))
            otr = max(0.1, np.random.normal(otr_m, otr_s))
            
            r_tm1 = np.random.normal(tm1, 0.5)
            r_tm2 = np.random.normal(tm2, 0.8)
            r_tm3 = np.random.normal(tm3, 0.8)
            r_tg  = np.random.normal(tg, 0.3)
            r_tc  = np.random.normal(tc, 0.4)
            r_tmax= np.random.normal(tmax, 1.0)
            
            rows.append(list(feat) + [ts, mod, el, wvp, otr, r_tm1, r_tm2, r_tm3, r_tg, r_tc, r_tmax])
            
    columns = ['GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness', 
               'Tensile_Strength', 'Modulus', 'Elongation', 'WVP', 'OTR',
               'Tm1', 'Tm2', 'Tm3', 'Tg', 'Tc', 'Tmax']
    return pd.DataFrame(rows, columns=columns)

df_synthetic = generate_monte_carlo_dataset()

# ==========================================
# 3. METAMODEL MACHINE LEARNING TRAIN ENGINE
# ==========================================
@st.cache_resource
def train_metamodels(df):
    features = ['GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness']
    targets = ['Tensile_Strength', 'Modulus', 'Elongation', 'WVP', 'OTR', 'Tm1', 'Tm2', 'Tm3', 'Tg', 'Tc', 'Tmax']
    
    X = df[features]
    trained_models = {}
    for target in targets:
        rf = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, df[target])
        trained_models[target] = rf
        
    return trained_models

models = train_metamodels(df_synthetic)

# ==========================================
# 4. STREAMLIT APP UI INTERFACE
# ==========================================
st.title("🔬 Biopolymer Film Crosslinking Metamodel Studio")
st.markdown("---")

st.sidebar.header("🛠️ Matrix Formulation Control")
gel = st.sidebar.slider("Gelatin Content (GEL %)", 0.0, 8.0, 4.0, 0.5)
rf = st.sidebar.slider("Riboflavin Content (RF %)", 0.0, 1.25, 1.25, 0.25)
obe = st.sidebar.slider("Bioactive Extract (OBE %)", 0.0, 5.0, 0.0, 1.0)
thickness = st.sidebar.slider("Film Thickness (mm)", 0.20, 0.35, 0.25, 0.01)

st.sidebar.header("⚡ Irradiation Treatment Configuration")
rad_selection = st.sidebar.selectbox("Radiation Type Source", ["UV-C (J/cm²)", "X-Ray (kGy)"])
is_xray = 1.0 if "X-Ray" in rad_selection else 0.0
dose = st.sidebar.slider("Active Exposure Dose Level", 0.0, 11.0, 4.0, 0.1)

current_query = pd.DataFrame([[gel, rf, obe, is_xray, dose, thickness]], 
                             columns=['GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness'])

# FIXED: Extracting the scalar using [0] allows live updates
predictions = {}
for target, model in models.items():
    predictions[target] = float(model.predict(current_query)[0])

tab1, tab2, tab3 = st.tabs(["📊 Live Parameter Engine", "🌌 3D Optimization Surface Mesh", "🧀 Cheese Packaging Feasibility"])

with tab1:
    st.subheader("🔮 Metamodel Real-time Prediction Dashboard")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("⚓ Mechanical Metrics")
        st.metric("Tensile Strength", f"{predictions['Tensile_Strength']:.2f} MPa")
        st.metric("Elastic Modulus", f"{predictions['Modulus']:.2f} MPa")
        st.metric("Elongation at Break", f"{predictions['Elongation']:.1f} %")
        
    with col2:
        st.warning("🛡️ Gas & Vapor Barrier Metrics")
        st.metric("Water Vapor Permeability (WVP)", f"{predictions['WVP']:.3f}")
        st.metric("Oxygen Transmission Rate (OTR)", f"{predictions['OTR']:.2f} cc/m².day")
        
    with col3:
        st.success("🔥 Thermal Characterization")
        st.metric("Melting Phase (Tm1 / Tm3)", f"{predictions['Tm1']:.1f}°C / {predictions['Tm3']:.1f}°C")
        st.metric("Crystallisation Temp (Tc)", f"{predictions['Tc']:.1f}°C")
        st.metric("Peak Breakdown (Tmax)", f"{predictions['Tmax']:.1f}°C")

with tab2:
    st.subheader("🌌 Multi-Variable 3D Optimization Space")
    
    plot_target = st.selectbox("Select Optimization Target Property Grid", 
                               ['Elongation', 'OTR', 'Tensile_Strength', 'WVP', 'Tmax'])
    
    dose_range = np.linspace(0, 11, 20)
    gel_range = np.linspace(0, 8, 20)
    D, G = np.meshgrid(dose_range, gel_range)
    
    mesh_rows = []
    for g_val, d_val in zip(G.ravel(), D.ravel()):
        mesh_rows.append([g_val, rf, obe, is_xray, d_val, thickness])
        
    mesh_df = pd.DataFrame(mesh_rows, columns=['GEL', 'RF', 'OBE', 'Is_Xray', 'Dose', 'Thickness'])
    mesh_preds = models[plot_target].predict(mesh_df).reshape(20, 20)
    
    fig = go.Figure(data=[go.Surface(z=mesh_preds, x=dose_range, y=gel_range, colorscale='Viridis')])
    fig.update_layout(
        title=f"3D Interaction Model Matrix ({plot_target})",
        scene=dict(xaxis_title="Radiation Dose Parameter", yaxis_title="Gelatin Content (%)", zaxis_title=plot_target),
        width=800, height=600
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🧀 Target Assessment: Premium Cheese Packaging Suitability")
    st.markdown("Target Parameters: **Elongation at Break > 200%** (Flexibility) and **OTR < 20 cc/m².day** (Oxidative Protection).")
    
    # FIXED: Clean evaluation variables that instantly trigger dynamic changes
    elong_ok = predictions['Elongation'] >= 200.0
    otr_ok = predictions['OTR'] <= 20.0
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Structural Elasticity Check (>200%):** {'🟢 PASSED' if elong_ok else '🔴 FAILED'} ({predictions['Elongation']:.1f}%)")
        st.markdown(f"**Oxygen Blockade Check (<20 cc):** {'🟢 PASSED' if otr_ok else '🔴 FAILED'} ({predictions['OTR']:.2f} cc/m².day)")
        
    with c2:
        if elong_ok and otr_ok:
            st.success("🏆 PERFECT FORMULATION: This functional boundary layout fits high-performance industrial cheese protection specs.")
        else:
            st.error("⚠️ SUB-OPTIMAL BOUNDARY: Crosslinking density or structural additions fail packaging compliance standards at this point.")

    st.markdown("### 🔍 Recommended Processing Windows")
    st.write("Below are the grouped formulation settings evaluated from the dataset. Rows are sorted to show optimal oxygen blocks first:")
    
    # FIXED: Group by formulation first, THEN filter the aggregates so high-yield windows appear cleanly
    grouped_runs = df_synthetic.groupby(['Is_Xray', 'Dose', 'GEL', 'RF', 'OBE']).mean().reset_index()
    
    # Filter using relaxed, realistic criteria to prevent an empty display grid
