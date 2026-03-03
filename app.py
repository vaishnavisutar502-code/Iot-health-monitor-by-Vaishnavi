import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Health Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@600;800&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .stApp { background: #050c14; color: #e2eaf5; }

  .metric-card {
    background: #0a1628;
    border: 1px solid #1a2d4a;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }
  .metric-card.bpm::before  { background: linear-gradient(90deg, #ff6b35, #ff9800); }
  .metric-card.spo2::before { background: linear-gradient(90deg, #00e5ff, #0080ff); }
  .metric-card.status::before { background: linear-gradient(90deg, #00ff9d, #00c060); }

  .metric-val  { font-size: 48px; font-weight: 800; line-height: 1; margin: 8px 0; }
  .metric-unit { font-family: 'Space Mono', monospace; font-size: 12px; color: #5a7a9a; letter-spacing: 2px; }
  .metric-label{ font-family: 'Space Mono', monospace; font-size: 11px; color: #5a7a9a; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 4px; }

  .bpm  .metric-val  { color: #ff6b35; }
  .spo2 .metric-val  { color: #00e5ff; }
  .status .metric-val { font-size: 24px; color: #00ff9d; }

  .alert-normal   { background: rgba(0,255,157,0.08);  border: 1px solid rgba(0,255,157,0.3);  border-radius: 10px; padding: 12px 18px; color: #00ff9d; }
  .alert-warning  { background: rgba(255,193,7,0.08);  border: 1px solid rgba(255,193,7,0.3);   border-radius: 10px; padding: 12px 18px; color: #ffc107; }
  .alert-critical { background: rgba(255,59,48,0.1);   border: 1px solid rgba(255,59,48,0.4);   border-radius: 10px; padding: 12px 18px; color: #ff3b30; }

  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #5a7a9a;
    text-transform: uppercase;
    margin: 28px 0 12px;
    border-left: 3px solid #00e5ff;
    padding-left: 10px;
  }

  div[data-testid="stSidebar"] { background: #070f1c; border-right: 1px solid #1a2d4a; }
  div[data-testid="stSidebar"] .stSelectbox label,
  div[data-testid="stSidebar"] .stTextInput label { color: #5a7a9a; font-size: 12px; }
  .stSelectbox > div > div { background: #0a1628 !important; border-color: #1a2d4a !important; color: #e2eaf5 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ IoT Health Monitor")
    st.markdown("---")

    st.markdown("### 🔑 ThingSpeak Config")
    CHANNEL_ID = st.text_input("Channel ID", value="3244796")
    READ_API   = st.text_input("Read API Key", value="6QLIS15N3QQTV21C", type="password")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    num_records    = st.slider("Records to fetch", 50, 500, 200, 50)
    refresh_sec    = st.selectbox("Auto-refresh every", [10, 15, 30, 60], index=1)
    selected_model = st.selectbox("ML Model", ["KNN", "Logistic Regression", "Both"])

    st.markdown("---")
    st.markdown("### 🚨 Alert Thresholds")
    bpm_low  = st.number_input("BPM Low  (Bradycardia)", value=60)
    bpm_high = st.number_input("BPM High (Tachycardia)", value=100)
    spo2_low = st.number_input("SpO₂ Low (Hypoxia %)",   value=94)

    st.markdown("---")
    auto_refresh = st.toggle("🔄 Auto Refresh", value=True)
    st.markdown(f"<p style='font-family:monospace;font-size:10px;color:#5a7a9a'>Refresh: every {refresh_sec}s</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  THINGSPEAK DATA FETCH
# ─────────────────────────────────────────────
@st.cache_data(ttl=15)
def fetch_thingspeak(channel_id: str, api_key: str, results: int = 200):
    """Fetch latest records from ThingSpeak channel."""
    url = (
        f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
        f"?api_key={api_key}&results={results}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        feeds = data.get("feeds", [])
        if not feeds:
            return None, "No data in channel"

        df = pd.DataFrame(feeds)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["bpm"]  = pd.to_numeric(df.get("field1"), errors="coerce")
        df["spo2"] = pd.to_numeric(df.get("field2"), errors="coerce")
        df = df.dropna(subset=["bpm", "spo2"]).reset_index(drop=True)
        df = df.sort_values("created_at").reset_index(drop=True)
        return df, None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to ThingSpeak. Check internet."
    except requests.exceptions.HTTPError as e:
        return None, f"❌ HTTP Error: {e}"
    except Exception as e:
        return None, f"❌ Error: {e}"


# ─────────────────────────────────────────────
#  AUTO-LABEL FUNCTION (rule-based for training)
# ─────────────────────────────────────────────
def label_health(bpm, spo2):
    if spo2 < 90:
        return "Critical Hypoxia"
    elif spo2 < 94:
        return "Hypoxia"
    elif bpm > 120:
        return "Severe Tachycardia"
    elif bpm > 100:
        return "Tachycardia"
    elif bpm < 50:
        return "Severe Bradycardia"
    elif bpm < 60:
        return "Bradycardia"
    else:
        return "Normal"

LABEL_COLOR = {
    "Normal":             "#00ff9d",
    "Tachycardia":        "#ffc107",
    "Severe Tachycardia": "#ff6b35",
    "Bradycardia":        "#00e5ff",
    "Severe Bradycardia": "#0080ff",
    "Hypoxia":            "#f59e0b",
    "Critical Hypoxia":   "#ff3b30",
}


# ─────────────────────────────────────────────
#  ML TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train KNN and LR on labelled ThingSpeak data."""
    df = df.copy()
    df["label"] = df.apply(lambda r: label_health(r["bpm"], r["spo2"]), axis=1)

    X = df[["bpm", "spo2"]].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Augment if too few samples
    if len(X_scaled) < 20:
        np.random.seed(42)
        noise = np.random.normal(0, 0.5, X_scaled.shape)
        X_scaled = np.vstack([X_scaled] * 5 + [X_scaled + noise])
        y = np.tile(y, 6)

    knn = KNeighborsClassifier(n_neighbors=min(5, len(X_scaled)-1))
    knn.fit(X_scaled, y)

    lr = LogisticRegression(max_iter=1000, multi_class='auto')
    lr.fit(X_scaled, y)

    return knn, lr, scaler, df


def predict(model, scaler, bpm_val, spo2_val):
    X = scaler.transform([[bpm_val, spo2_val]])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf  = max(proba) * 100
    return pred, conf


# ─────────────────────────────────────────────
#  GAUGE CHART
# ─────────────────────────────────────────────
def make_gauge(value, title, min_val, max_val, color, unit=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": unit, "font": {"size": 36, "color": color}},
        title={"text": title, "font": {"size": 13, "color": "#5a7a9a", "family": "Space Mono"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#1a2d4a", "tickfont": {"color": "#5a7a9a", "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor":   "#0a1628",
            "bordercolor": "#1a2d4a",
            "steps": [
                {"range": [min_val, max_val * 0.4], "color": "rgba(255,255,255,0.03)"},
                {"range": [max_val * 0.4, max_val * 0.8], "color": "rgba(255,255,255,0.05)"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "value": value}
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="#0a1628",
        font_color="#e2eaf5",
    )
    return fig


# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
st.markdown("<h1 style='font-size:32px;font-weight:800;margin-bottom:4px'>❤️ IoT Health Monitoring Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family:Space Mono,monospace;font-size:12px;color:#5a7a9a;margin-bottom:20px'>MAX30102 + Arduino Mega 2560 → ThingSpeak → ML Classification</p>", unsafe_allow_html=True)

# Validate config
if not CHANNEL_ID.strip():
    st.warning("👈 Enter your **ThingSpeak Channel ID** in the sidebar to get started.")
    st.info("Your Channel ID is visible in the URL: `thingspeak.com/channels/YOUR_CHANNEL_ID`")
    st.stop()

# Fetch data
df, err = fetch_thingspeak(CHANNEL_ID.strip(), READ_API.strip(), num_records)

if err:
    st.error(err)
    st.stop()

if df is None or df.empty:
    st.warning("No valid data found in ThingSpeak channel. Make sure Field1=BPM and Field2=SpO₂.")
    st.stop()

# Latest reading
latest    = df.iloc[-1]
latest_bpm  = float(latest["bpm"])
latest_spo2 = float(latest["spo2"])
latest_time = latest["created_at"].strftime("%H:%M:%S  %d %b %Y")
health_rule = label_health(latest_bpm, latest_spo2)
health_color= LABEL_COLOR.get(health_rule, "#e2eaf5")

# ── LIVE METRICS ROW ──
st.markdown('<div class="section-title">LIVE READING</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card bpm">
      <div class="metric-label">Heart Rate</div>
      <div class="metric-val">{latest_bpm:.0f}</div>
      <div class="metric-unit">BPM</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card spo2">
      <div class="metric-label">Blood Oxygen</div>
      <div class="metric-val">{latest_spo2:.1f}</div>
      <div class="metric-unit">SpO₂ %</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card status">
      <div class="metric-label">Health Status</div>
      <div class="metric-val" style="color:{health_color}">{health_rule}</div>
      <div class="metric-unit">RULE-BASED</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card" style="background:#0a1628;border:1px solid #1a2d4a;border-radius:14px;padding:20px 24px;text-align:center;">
      <div class="metric-label">Last Updated</div>
      <div style="font-size:15px;font-weight:700;color:#e2eaf5;margin:12px 0">{latest_time}</div>
      <div class="metric-unit">📡 {len(df)} records loaded</div>
    </div>""", unsafe_allow_html=True)

# ── ALERT BANNER ──
st.markdown('<div class="section-title">ALERTS</div>', unsafe_allow_html=True)
alert_msgs = []
if latest_bpm > bpm_high:
    alert_msgs.append(f"⚡ High Heart Rate: {latest_bpm:.0f} BPM (Normal < {bpm_high})")
if latest_bpm < bpm_low:
    alert_msgs.append(f"🔵 Low Heart Rate: {latest_bpm:.0f} BPM (Normal > {bpm_low})")
if latest_spo2 < spo2_low:
    alert_msgs.append(f"🔴 Low Blood Oxygen: {latest_spo2:.1f}% (Normal > {spo2_low}%)")

if alert_msgs:
    for msg in alert_msgs:
        st.markdown(f'<div class="alert-critical">🚨 {msg}</div>', unsafe_allow_html=True)
    st.markdown("")
else:
    st.markdown('<div class="alert-normal">✅ All vitals are within normal range.</div>', unsafe_allow_html=True)

st.markdown("")

# ── GAUGES ──
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(make_gauge(latest_bpm, "HEART RATE (BPM)", 30, 180, "#ff6b35", " BPM"), use_container_width=True)
with g2:
    st.plotly_chart(make_gauge(latest_spo2, "BLOOD OXYGEN (SpO₂)", 80, 100, "#00e5ff", "%"), use_container_width=True)

# ── HISTORICAL CHARTS ──
st.markdown('<div class="section-title">HISTORICAL TRENDS</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 Time Series", "📊 Distribution"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["created_at"], y=df["bpm"],
        name="BPM", line=dict(color="#ff6b35", width=2),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.07)"
    ))
    fig.add_trace(go.Scatter(
        x=df["created_at"], y=df["spo2"],
        name="SpO₂ %", line=dict(color="#00e5ff", width=2),
        yaxis="y2", fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"
    ))
    fig.add_hline(y=bpm_high, line_dash="dot", line_color="rgba(255,107,53,0.4)", annotation_text="BPM High")
    fig.add_hline(y=bpm_low,  line_dash="dot", line_color="rgba(0,229,255,0.4)",  annotation_text="BPM Low")
    fig.update_layout(
        height=360,
        paper_bgcolor="#0a1628", plot_bgcolor="#050c14",
        font_color="#5a7a9a",
        legend=dict(bgcolor="#0a1628", bordercolor="#1a2d4a"),
        xaxis=dict(gridcolor="#1a2d4a", showgrid=True),
        yaxis=dict(title="BPM", gridcolor="#1a2d4a", color="#ff6b35"),
        yaxis2=dict(title="SpO₂ %", overlaying="y", side="right", color="#00e5ff", range=[80, 102]),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_h = px.histogram(df, x="bpm", nbins=20, color_discrete_sequence=["#ff6b35"],
                              title="BPM Distribution")
        fig_h.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#050c14",
                             font_color="#5a7a9a", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    with col_b:
        fig_h2 = px.histogram(df, x="spo2", nbins=20, color_discrete_sequence=["#00e5ff"],
                               title="SpO₂ Distribution")
        fig_h2.update_layout(paper_bgcolor="#0a1628", plot_bgcolor="#050c14",
                              font_color="#5a7a9a", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h2, use_container_width=True)

# ── ML PREDICTION ──
st.markdown('<div class="section-title">ML PREDICTION ENGINE</div>', unsafe_allow_html=True)

with st.spinner("Training ML models on ThingSpeak data..."):
    knn, lr, scaler, df_labeled = train_models(df)

col_ml1, col_ml2 = st.columns([1, 2])

with col_ml1:
    st.markdown("**Predict for current reading:**")
    inp_bpm  = st.number_input("BPM Input",  value=float(latest_bpm),  step=1.0)
    inp_spo2 = st.number_input("SpO₂ Input", value=float(latest_spo2), step=0.1)
    predict_btn = st.button("🔮 Run Prediction", use_container_width=True)

with col_ml2:
    if predict_btn or True:   # always show latest
        results = {}
        if selected_model in ["KNN", "Both"]:
            pred_k, conf_k = predict(knn, scaler, inp_bpm, inp_spo2)
            results["KNN"] = (pred_k, conf_k)
        if selected_model in ["Logistic Regression", "Both"]:
            pred_l, conf_l = predict(lr,  scaler, inp_bpm, inp_spo2)
            results["Logistic Regression"] = (pred_l, conf_l)

        for model_name, (pred, conf) in results.items():
            color = LABEL_COLOR.get(pred, "#e2eaf5")
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #1a2d4a;border-left:4px solid {color};
                        border-radius:10px;padding:16px;margin-bottom:12px">
              <div style="font-family:Space Mono,monospace;font-size:10px;color:#5a7a9a;letter-spacing:2px">{model_name.upper()}</div>
              <div style="font-size:22px;font-weight:800;color:{color};margin:6px 0">{pred}</div>
              <div style="font-family:Space Mono,monospace;font-size:11px;color:#5a7a9a">
                Confidence: <span style="color:{color}">{conf:.1f}%</span>
              </div>
              <div style="background:#1a2d4a;border-radius:20px;height:6px;margin-top:10px">
                <div style="background:{color};width:{conf:.0f}%;height:6px;border-radius:20px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── MODEL EVALUATION ──
st.markdown('<div class="section-title">MODEL EVALUATION</div>', unsafe_allow_html=True)

df_labeled["knn_pred"] = knn.predict(scaler.transform(df_labeled[["bpm", "spo2"]].values))
df_labeled["lr_pred"]  = lr.predict( scaler.transform(df_labeled[["bpm", "spo2"]].values))
knn_acc = accuracy_score(df_labeled["label"], df_labeled["knn_pred"]) * 100
lr_acc  = accuracy_score(df_labeled["label"], df_labeled["lr_pred"])  * 100

e1, e2 = st.columns(2)
with e1:
    fig_acc = go.Figure(go.Bar(
        x=["KNN", "Logistic Regression"],
        y=[knn_acc, lr_acc],
        marker_color=["#ff6b35", "#a855f7"],
        text=[f"{knn_acc:.1f}%", f"{lr_acc:.1f}%"],
        textposition="outside",
        textfont=dict(color="#e2eaf5")
    ))
    fig_acc.update_layout(
        title="Model Accuracy (%)",
        paper_bgcolor="#0a1628", plot_bgcolor="#050c14",
        font_color="#5a7a9a", yaxis=dict(range=[0, 110], gridcolor="#1a2d4a"),
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with e2:
    label_dist = df_labeled["label"].value_counts().reset_index()
    label_dist.columns = ["label", "count"]
    colors = [LABEL_COLOR.get(l, "#e2eaf5") for l in label_dist["label"]]
    fig_pie = go.Figure(go.Pie(
        labels=label_dist["label"], values=label_dist["count"],
        marker_colors=colors, hole=0.5
    ))
    fig_pie.update_layout(
        title="Health Status Distribution",
        paper_bgcolor="#0a1628", font_color="#5a7a9a",
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── DATA TABLE ──
st.markdown('<div class="section-title">RAW DATA TABLE</div>', unsafe_allow_html=True)
with st.expander("View & Download Data"):
    display_df = df_labeled[["created_at", "bpm", "spo2", "label"]].copy()
    display_df.columns = ["Timestamp", "BPM", "SpO₂ %", "Health Label"]
    display_df = display_df.sort_values("Timestamp", ascending=False)
    st.dataframe(display_df, use_container_width=True, height=280)
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "health_data.csv", "text/csv")

# ── FOOTER & AUTO REFRESH ──
st.markdown("---")
st.markdown(
    "<p style='font-family:Space Mono,monospace;font-size:10px;color:#1a2d4a;text-align:center'>"
    "IoT Health Monitor • MAX30102 + Arduino Mega 2560 + ThingSpeak + Streamlit</p>",
    unsafe_allow_html=True
)

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
