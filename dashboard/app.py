from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Social Media Impact on Students | AI Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics, glassmorphism, and responsive cards
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Hero Container */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(49, 46, 129, 0.8) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);
        color: #f8fafc;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -6px rgba(56, 189, 248, 0.25);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Result Badges */
    .result-badge {
        padding: 18px 24px;
        border-radius: 16px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .badge-beneficial {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .badge-neutral {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.3) 100%);
        border: 1px solid #3b82f6;
        color: #60a5fa;
    }
    .badge-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.3) 100%);
        border: 1px solid #ef4444;
        color: #f87171;
    }

    /* Platform Pills */
    .platform-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper function for width parameter (handles newer and older streamlit versions)
def get_width_kwarg(val="stretch"):
    return {"width": val}

# ==========================================
# Load Model Pipeline
# ==========================================
MODEL_PATHS = [
    Path(__file__).resolve().parent.parent / "models" / "Social_media_predector.pkl",
    Path("models") / "Social_media_predector.pkl",
    Path("app") / "Social_media_predector.pkl",
    Path("Social_media_predector.pkl"),
    Path("/code/models") / "Social_media_predector.pkl",
]


@st.cache_resource(show_spinner=False)
def load_ml_pipeline():
    for p in MODEL_PATHS:
        if p.exists() and p.is_file():
            try:
                loaded = joblib.load(p)
                return loaded, str(p)
            except Exception as e:
                st.error(f"Error loading model from {p}: {e}")
    return None, None


model, model_source_path = load_ml_pipeline()

LABEL_MAP = {
    0: ("Neutral", "badge-neutral", "⚖️", "Social media has a moderate or balanced impact on this student's academic and emotional routine."),
    1: ("Beneficial", "badge-beneficial", "🌟", "Social media provides positive support, educational motivation, or healthy social connectivity."),
    2: ("Negative", "badge-negative", "⚠️", "Social media is noticeably linked with higher stress, poor sleep patterns, or academic fatigue."),
}

# ==========================================
# Benchmark Comparison Data
# ==========================================
BENCHMARK_DATA = [
    {"Model": "Logistic Regression", "Accuracy": 0.9856, "Precision": 0.9672, "Recall": 0.9451, "F1-Score": 0.9554},
    {"Model": "KNN", "Accuracy": 0.9556, "Precision": 0.9255, "Recall": 0.8717, "F1-Score": 0.8969},
    {"Model": "KNN GridSearch", "Accuracy": 0.9489, "Precision": 0.9033, "Recall": 0.8552, "F1-Score": 0.8778},
    {"Model": "Decision Tree", "Accuracy": 0.9711, "Precision": 0.9143, "Recall": 0.9224, "F1-Score": 0.9183},
    {"Model": "Decision Tree GridSearch", "Accuracy": 0.9733, "Precision": 0.9494, "Recall": 0.9510, "F1-Score": 0.9500},
    {"Model": "Random Forest", "Accuracy": 0.9778, "Precision": 0.9561, "Recall": 0.9356, "F1-Score": 0.9451},
    {"Model": "Random Forest GridSearch", "Accuracy": 0.9733, "Precision": 0.9494, "Recall": 0.9510, "F1-Score": 0.9500},
    {"Model": "XGBoost", "Accuracy": 0.9867, "Precision": 0.9664, "Recall": 0.9593, "F1-Score": 0.9624},
    {"Model": "XGBoost GridSearch", "Accuracy": 0.9456, "Precision": 0.9494, "Recall": 0.7695, "F1-Score": 0.8374},
    {"Model": "Voting Classifier", "Accuracy": 0.9833, "Precision": 0.9533, "Recall": 0.9538, "F1-Score": 0.9534},
]
df_benchmark = pd.DataFrame(BENCHMARK_DATA)

# ==========================================
# Header / Hero Section
# ==========================================
st.markdown(
    """
    <div class="hero-card">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <span style="font-size: 2.5rem;">📱⚡🎓</span>
            <div>
                <div class="hero-title">Social Media Impact on Students</div>
                <div class="hero-subtitle">
                    Intelligent Machine Learning System analyzing social media habits, sleep patterns, stress levels, and GPA to evaluate student well-being.
                </div>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <span class="platform-pill" style="border-color:#E1306C; color:#f43f5e;">📸 Instagram</span>
            <span class="platform-pill" style="border-color:#25F4EE; color:#06b6d4;">🎵 TikTok</span>
            <span class="platform-pill" style="border-color:#FF0000; color:#ef4444;">▶️ YouTube</span>
            <span class="platform-pill" style="border-color:#FFFC00; color:#eab308;">👻 Snapchat</span>
            <span class="platform-pill" style="border-color:#0077B5; color:#38bdf8;">💼 LinkedIn</span>
            <span class="platform-pill" style="border-color:#FF4500; color:#fb923c;">👾 Reddit</span>
            <span class="platform-pill" style="border-color:#ffffff; color:#e2e8f0;">✖️ X (Twitter)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPIs Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown('<div class="metric-box"><div class="metric-value">98.67%</div><div class="metric-label">Best Model Accuracy (XGBoost)</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="metric-box"><div class="metric-value">0.9624</div><div class="metric-label">Top F1-Score (Macro)</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="metric-box"><div class="metric-value">14</div><div class="metric-label">Engineered Features</div></div>', unsafe_allow_html=True)
with kpi4:
    status_text = "Ready 🟢" if model is not None else "Missing Model 🔴"
    st.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size:1.3rem;">{status_text}</div><div class="metric-label">ML Pipeline Status</div></div>', unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_predict, tab_benchmark, tab_insights = st.tabs([
    "🔮 Predict Student Impact",
    "🏆 Model Performance & Comparison",
    "📈 Feature Analytics & About",
])

# ==========================================
# TAB 1: PREDICTION INTERFACE
# ==========================================
with tab_predict:
    st.subheader("Student Profile & Usage Form")
    st.caption("Fill in student details below or choose a quick sample profile to simulate immediate predictions.")

    # Quick Sample Preset Buttons
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    preset_data = None
    if sample_col1.button("✨ Load Balanced Student (Neutral/Positive)", **get_width_kwarg("stretch")):
        preset_data = {
            "Age": 20, "Gender": "Female", "Academic_Level": "Undergraduate",
            "Primary_Platform": "YouTube", "Daily_Usage_Hours": 2.5, "Weekend_Extra_Hours": 1.0,
            "Device_Type": "Laptop/PC", "Sleep_Duration_Hours": 7.5, "Sleep_Quality_Score": 4,
            "Late_Night_Usage": 0, "Social_Comparison_Frequency": "Rarely",
            "Perceived_Stress_Score": 12.0, "Mental_Health_Index": 75, "Academic_Performance_GPA": 3.6
        }
    if sample_col2.button("⚠️ Load High-Stress / Late-Night User (Negative)", **get_width_kwarg("stretch")):
        preset_data = {
            "Age": 19, "Gender": "Male", "Academic_Level": "Undergraduate",
            "Primary_Platform": "TikTok", "Daily_Usage_Hours": 7.5, "Weekend_Extra_Hours": 3.5,
            "Device_Type": "Smartphone", "Sleep_Duration_Hours": 4.5, "Sleep_Quality_Score": 1,
            "Late_Night_Usage": 1, "Social_Comparison_Frequency": "Always",
            "Perceived_Stress_Score": 34.0, "Mental_Health_Index": 35, "Academic_Performance_GPA": 2.2
        }
    if sample_col3.button("📚 Load Career / Learning Focused (Beneficial)", **get_width_kwarg("stretch")):
        preset_data = {
            "Age": 23, "Gender": "Female", "Academic_Level": "Postgraduate",
            "Primary_Platform": "LinkedIn", "Daily_Usage_Hours": 1.5, "Weekend_Extra_Hours": 0.5,
            "Device_Type": "Laptop/PC", "Sleep_Duration_Hours": 8.0, "Sleep_Quality_Score": 5,
            "Late_Night_Usage": 0, "Social_Comparison_Frequency": "Never",
            "Perceived_Stress_Score": 7.0, "Mental_Health_Index": 88, "Academic_Performance_GPA": 3.9
        }

    # Initialize session state for presets
    if preset_data:
        st.session_state.student_inputs = preset_data

    inputs = st.session_state.get("student_inputs", {
        "Age": 20,
        "Gender": "Male",
        "Academic_Level": "Undergraduate",
        "Primary_Platform": "Instagram",
        "Daily_Usage_Hours": 6.5,
        "Weekend_Extra_Hours": 2.0,
        "Device_Type": "Smartphone",
        "Sleep_Duration_Hours": 6.0,
        "Sleep_Quality_Score": 5,
        "Late_Night_Usage": 1,
        "Social_Comparison_Frequency": "Frequently",
        "Perceived_Stress_Score": 8.0,
        "Mental_Health_Index": 45,
        "Academic_Performance_GPA": 2.7,
    })

    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("#### 👤 Demographics")
            age = st.slider("Age", min_value=15, max_value=30, value=int(inputs["Age"]))
            gender_options = ["Female", "Male", "Non-Binary", "Prefer not to say"]
            gender = st.selectbox("Gender", gender_options, index=gender_options.index(inputs["Gender"]) if inputs["Gender"] in gender_options else 0)
            acad_options = ["Undergraduate", "High School", "Postgraduate"]
            academic_level = st.selectbox("Academic Level", acad_options, index=acad_options.index(inputs["Academic_Level"]) if inputs["Academic_Level"] in acad_options else 0)
            gpa = st.slider("Academic GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=float(inputs["Academic_Performance_GPA"]), step=0.05)

        with col_b:
            st.markdown("#### 📱 Social Media Usage")
            platform_options = ["Instagram", "TikTok", "YouTube", "Snapchat", "LinkedIn", "Reddit", "X (Twitter)"]
            platform = st.selectbox("Primary Platform", platform_options, index=platform_options.index(inputs["Primary_Platform"]) if inputs["Primary_Platform"] in platform_options else 0)
            daily_hours = st.slider("Daily Usage (Hours)", min_value=0.0, max_value=16.0, value=float(inputs["Daily_Usage_Hours"]), step=0.5)
            weekend_hours = st.slider("Weekend Extra Hours", min_value=0.0, max_value=8.0, value=float(inputs["Weekend_Extra_Hours"]), step=0.5)
            device_options = ["Smartphone", "Tablet", "Laptop/PC"]
            device = st.selectbox("Primary Device", device_options, index=device_options.index(inputs["Device_Type"]) if inputs["Device_Type"] in device_options else 0)

        with col_c:
            st.markdown("#### 🧠 Sleep & Well-being")
            sleep_hours = st.slider("Sleep Duration (Hours)", min_value=2.0, max_value=12.0, value=float(inputs["Sleep_Duration_Hours"]), step=0.5)
            sleep_quality = st.select_slider("Sleep Quality Score", options=[1, 2, 3, 4, 5], value=int(inputs["Sleep_Quality_Score"]))
            late_night = st.radio("Late Night Social Usage?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=1 if inputs["Late_Night_Usage"] == 1 else 0)[1]
            comp_options = ["Never", "Rarely", "Sometimes", "Frequently", "Always"]
            comparison_freq = st.selectbox("Social Comparison Frequency", comp_options, index=comp_options.index(inputs["Social_Comparison_Frequency"]) if inputs["Social_Comparison_Frequency"] in comp_options else 2)
            stress_score = st.slider("Perceived Stress Score (0 - 40)", min_value=0.0, max_value=40.0, value=float(inputs["Perceived_Stress_Score"]), step=1.0)
            mental_health = st.slider("Mental Health Index (0 - 100)", min_value=0, max_value=100, value=int(inputs["Mental_Health_Index"]))

        submit_btn = st.form_submit_button("🚀 Run AI Impact Assessment", type="primary", **get_width_kwarg("stretch"))

    if submit_btn:
        if model is None:
            st.error("Model file `Social_media_predector.pkl` could not be located or loaded.")
        else:
            # Construct DataFrame exactly matching pipeline expectations
            input_dict = {
                "Age": [age],
                "Gender": [gender],
                "Academic_Level": [academic_level],
                "Primary_Platform": [platform],
                "Daily_Usage_Hours": [daily_hours],
                "Weekend_Extra_Hours": [weekend_hours],
                "Device_Type": [device],
                "Sleep_Duration_Hours": [sleep_hours],
                "Sleep_Quality_Score": [sleep_quality],
                "Late_Night_Usage": [int(late_night)],
                "Social_Comparison_Frequency": [comparison_freq],
                "Perceived_Stress_Score": [stress_score],
                "Mental_Health_Index": [mental_health],
                "Academic_Performance_GPA": [gpa],
            }
            sample_df = pd.DataFrame(input_dict)

            with st.spinner("Analyzing student metrics through XGBoost pipeline..."):
                pred_raw = model.predict(sample_df)[0]
                pred_class = int(pred_raw)
                probabilities = model.predict_proba(sample_df)[0] if hasattr(model, "predict_proba") else None

            label_name, badge_cls, icon, advice = LABEL_MAP.get(pred_class, ("Unknown", "badge-neutral", "❓", "No description"))

            st.write("---")
            st.markdown("### 🎯 Assessment Results")

            res_col1, res_col2 = st.columns([1.1, 1.4])

            with res_col1:
                st.markdown(
                    f"""
                    <div class="result-badge {badge_cls}">
                        <span style="font-size: 2rem;">{icon}</span>
                        <div>
                            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Predicted Impact</div>
                            <div style="font-size: 1.6rem; font-weight: 800;">{label_name} Impact</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.info(f"💡 **Diagnosis Insight:** {advice}")

                # Quick Behavioral Summary
                st.markdown("##### 📌 Student Highlights:")
                st.markdown(f"- **Screen Time:** `{daily_hours} hrs/day` (+{weekend_hours}h weekends)")
                st.markdown(f"- **Sleep vs Quality:** `{sleep_hours} hrs` (Score: `{sleep_quality}/5`)")
                st.markdown(f"- **Stress & Wellness:** Stress `{stress_score}/40` | MHI `{mental_health}/100`")
                st.markdown(f"- **Academic Standing:** GPA `{gpa}` ({academic_level})")

            with res_col2:
                if probabilities is not None:
                    prob_df = pd.DataFrame({
                        "Impact": ["Neutral", "Beneficial", "Negative"],
                        "Probability": probabilities,
                        "Color": ["#3b82f6", "#10b981", "#ef4444"]
                    })

                    fig_prob = go.Figure()
                    fig_prob.add_trace(go.Bar(
                        x=prob_df["Probability"],
                        y=prob_df["Impact"],
                        orientation='h',
                        marker=dict(
                            color=prob_df["Color"],
                            line=dict(color='rgba(255,255,255,0.2)', width=1)
                        ),
                        text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
                        textposition='auto',
                    ))
                    fig_prob.update_layout(
                        title="Model Confidence Distribution",
                        xaxis=dict(title="Probability", range=[0, 1], tickformat=".0%"),
                        yaxis=dict(autorange="reversed"),
                        height=260,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1"),
                    )
                    st.plotly_chart(fig_prob, **get_width_kwarg("stretch"))

# ==========================================
# TAB 2: BENCHMARK TABLE & CHARTS
# ==========================================
with tab_benchmark:
    st.subheader("Model Performance Benchmark")
    st.markdown(
        """
        Below is the benchmark evaluation of the trained classification models evaluated across 
        **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
        """
    )

    # Highlight top values
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: rgba(16, 185, 129, 0.25); color: #34d399; font-weight: bold;' if v else '' for v in is_max]

    styled_df = (
        df_benchmark.style
        .format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
        })
        .apply(highlight_best, subset=["Accuracy", "Precision", "Recall", "F1-Score"])
    )

    st.dataframe(styled_df, hide_index=False, **get_width_kwarg("stretch"))

    st.markdown("---")
    st.subheader("📊 Interactive Metric Visualizations")

    # Multi-metric grouped bar chart
    fig_models = px.bar(
        df_benchmark.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Comprehensive Performance Comparison across All Models",
        color_discrete_sequence=["#38bdf8", "#818cf8", "#a78bfa", "#34d399"],
        range_y=[0.70, 1.02]
    )
    fig_models.update_layout(
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
    )
    st.plotly_chart(fig_models, **get_width_kwarg("stretch"))

    # Best Model Callout Card
    best_row = df_benchmark.loc[df_benchmark["Accuracy"].idxmax()]
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(30, 41, 59, 0.8) 100%); border: 1px solid #10b981; border-radius: 16px; padding: 20px; margin-top: 15px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="background: #10b981; color: #022c22; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem;">SELECTED PRODUCTION MODEL</span>
                    <h3 style="margin: 8px 0 4px 0; color: #f8fafc;">🏆 {best_row['Model']}</h3>
                    <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">
                        Delivers the highest overall Accuracy ({best_row['Accuracy']*100:.2f}%) and F1-Score ({best_row['F1-Score']:.4f}) with optimal balance across all classes.
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.2rem; font-weight: 800; color: #34d399;">{best_row['Accuracy']*100:.2f}%</div>
                    <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Top Accuracy</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# TAB 3: INSIGHTS & ARCHITECTURE
# ==========================================
with tab_insights:
    st.subheader("💡 Key Dataset & Machine Learning Insights")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown(
            """
            #### 🔍 Determinants of Negative Impact:
            - **Late-Night Scrolling**: Strongly correlated with reduced sleep quality and impaired memory retention.
            - **Social Comparison Frequency**: Students who report 'Frequently' or 'Always' comparing their lifestyle to online peers show a 40%+ increase in perceived stress.
            - **Excessive Daily Usage (>6 hrs)**: Clear tipping point where academic performance drops and negative impacts become predominant.

            #### 🌿 Determinants of Beneficial Impact:
            - **Purpose-driven platforms**: YouTube & LinkedIn are predominantly associated with beneficial impact when usage is focused on career skills and tutorials.
            - **Moderate screen time (1.5 - 3.0 hrs/day)**: Keeps students socially connected without disrupting sleep hygiene.
            """
        )

    with col_i2:
        st.markdown(
            """
            #### ⚙️ ML Pipeline Architecture:
            - **Feature Preprocessing:**
              - `StandardScaler`: Applied to numeric metrics (Age, Usage Hours, Sleep, GPA, Stress, MHI).
              - `OneHotEncoder(drop='first', handle_unknown='ignore')`: Applied to categorical attributes.
              - `OrdinalEncoder`: Preserves sequence for `Social_Comparison_Frequency` (`Never` → `Always`).
              - `Passthrough`: Retains binary flag `Late_Night_Usage` (0/1).
            - **Model Estimator:** `XGBClassifier` with stratified multi-class objective (`Neutral: 0`, `Beneficial: 1`, `Negative: 2`).
            - **Saved Artifact:** Serialized via `joblib` into `Social_media_predector.pkl`.
            """
        )

    # Footer
    st.write("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>Social Media Impact on Students • AI Analytics Dashboard • Powered by Streamlit, Scikit-Learn & XGBoost</div>",
        unsafe_allow_html=True,
    )
