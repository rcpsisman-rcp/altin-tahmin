import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from arch import arch_model
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ============================================
# SAYFA AYARLARI
# ============================================
st.set_page_config(
    page_title="Altın Tahmin Platformu",
    page_icon="🥇",
    layout="wide"
)

# ============================================
# STİL
# ============================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #f0a500;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .tahmin-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# VERİ ÇEKME (cache ile hızlı)
# ============================================
@st.cache_data(ttl=3600)
def veri_cek():
    altin  = yf.download("GC=F",     start="2015-01-01")["Close"]
    bist   = yf.download("XU100.IS", start="2015-01-01")["Close"]
    petrol = yf.download("CL=F",     start="2015-01-01")["Close"]
    eurusd = yf.download("EURUSD=X", start="2015-01-01")["Close"]
    faiz   = yf.download("^TNX",     start="2015-01-01")["Close"]
    df = pd.concat([altin, bist, petrol, eurusd, faiz], axis=1)
    df.columns = ["Altin", "BIST100", "Petrol", "EURUSD", "Faiz"]
    return df.dropna()

@st.cache_data(ttl=3600)
def model_kur(df):
    log_df = np.log(df / df.shift(1)).dropna()
    var_model = VAR(log_df)
    lag = var_model.select_order(maxlags=10).aic
    var_fit = var_model.fit(lag)
    Y = log_df["Altin"]
    X = sm.add_constant(log_df[["BIST100", "Petrol", "EURUSD", "Faiz"]])
    ols = sm.OLS(Y, X).fit(cov_type="HC3")
    artik = ols.resid * 100
    garch = arch_model(artik, vol="Garch", p=1, q=1, dist="t").fit(disp="off")
    return log_df, var_fit, lag, ols, garch

def tahmin_yap(log_df, var_fit, lag, garch, gun):
    son = log_df.values[-lag:]
    tahmin = var_fit.forecast(son, steps=gun)
    tahmin_df = pd.DataFrame(tahmin, columns=log_df.columns)
    guncel = log_df["Altin"].iloc[-1]
    gercek_son_fiyat = float(np.exp(log_df["Altin"].cumsum().iloc[-1]))
    altin_son = yf.download("GC=F", period="5d")["Close"].iloc[-1]
    kumulatif = tahmin_df["Altin"].cumsum()
    fiyat = float(altin_son) * np.exp(kumulatif)
    son_vol = garch.conditional_volatility.iloc[-1] / 100
    std = son_vol * np.sqrt(np.arange(1, gun+1))
    ust = float(altin_son) * np.exp(kumulatif + 1.96 * std)
    alt = float(altin_son) * np.exp(kumulatif - 1.96 * std)
    tarihler = pd.date_range(start=pd.Timestamp.today(), periods=gun, freq="B")
    return fiyat.values, ust.values, alt.values, tarihler

# ============================================
# BAŞLIK
# ============================================
st.markdown("# 🥇 Altın Fiyat Tahmin Platformu")
st.markdown("*VAR + GARCH + OLS Ekonometrik Model | Gerçek Zamanlı Veri*")
st.divider()

# ============================================
# VERİ YÜKLEMESİ
# ============================================
with st.spinner("Veriler yükleniyor, model kuruluyor..."):
    df = veri_cek()
    log_df, var_fit, lag, ols, garch = model_kur(df)

guncel_fiyat = float(yf.download("GC=F", period="5d")["Close"].iloc[-1])
onceki_fiyat = float(yf.download("GC=F", period="5d")["Close"].iloc[-2])
degisim = ((guncel_fiyat - onceki_fiyat) / onceki_fiyat) * 100

# ============================================
# ÜST METRİKLER
# ============================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Güncel Altın Fiyatı", f"${guncel_fiyat:,.0f}", f"%{degisim:.2f}")
with col2:
    st.metric("📊 Model R²", f"%{ols.rsquared*100:.1f}", "OLS Robust")
with col3:
    alpha = garch.params["alpha[1]"]
    beta  = garch.params["beta[1]"]
    st.metric("📈 GARCH Kalıcılık", f"{alpha+beta:.4f}", "α+β")
with col4:
    st.metric("🔢 VAR Optimal Lag", f"{lag} gün", "AIC kriteri")

st.divider()

# ============================================
# TAHMİN SÜRESİ SEÇİMİ
# ============================================
st.markdown("## 📅 Tahmin Ufku Seçin")
sure_sec = st.radio(
    "",
    ["3 Ay", "6 Ay", "12 Ay", "2 Yıl"],
    horizontal=True
)

sure_gun = {"3 Ay": 63, "6 Ay": 126, "12 Ay": 252, "2 Yıl": 504}
gun = sure_gun[sure_sec]

with st.spinner(f"{sure_sec} tahmini hesaplanıyor..."):
    fiyat, ust, alt, tarihler = tahmin_yap(log_df, var_fit, lag, garch, gun)

son_tahmin = fiyat[-1]
son_ust    = ust[-1]
son_alt    = alt[-1]
pct        = ((son_tahmin - guncel_fiyat) / guncel_fiyat) * 100

# ============================================
# SENARYO KARTLARI
# ============================================
st.markdown("### 📊 Senaryo Analizi")
c1, c2, c3 = st.columns(3)
with c1:
    st.success(f"🟢 **Yükseliş Senaryosu**\n\n### ${son_ust:,.0f}\n\nFed faiz indirimi + dolar zayıflığı")
with c2:
    st.info(f"🟡 **Baz Senaryo (VAR Tahmini)**\n\n### ${son_tahmin:,.0f}\n\nMevcut makro ortamın devamı · %{pct:.1f}")
with c3:
    st.error(f"🔴 **Düşüş Senaryosu**\n\n### ${son_alt:,.0f}\n\nFed faiz artırımı + dolar rallisi")

st.divider()

# ============================================
# ANA TAHMİN GRAFİĞİ
# ============================================
st.markdown("### 📈 Fiyat Tahmini Grafiği")

gecmis_gun = 180
gecmis = df["Altin"].iloc[-gecmis_gun:]

fig = go.Figure()

# Geçmiş fiyat
fig.add_trace(go.Scatter(
    x=gecmis.index, y=gecmis.values,
    name="Gerçek Fiyat",
    line=dict(color="#f0a500", width=2)
))

# Güven aralığı
fig.add_trace(go.Scatter(
    x=list(tarihler) + list(tarihler[::-1]),
    y=list(ust) + list(alt[::-1]),
    fill="toself",
    fillcolor="rgba(0, 120, 255, 0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="%95 Güven Aralığı"
))

# Tahmin çizgisi
fig.add_trace(go.Scatter(
    x=tarihler, y=fiyat,
    name=f"VAR Tahmini ({sure_sec})",
    line=dict(color="#00d4ff", width=2.5, dash="dash")
))

# Bugün çizgisi
fig.add_vline(
    x=pd.Timestamp.today(),
    line_dash="dot",
    line_color="red",
    annotation_text="Bugün"
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    title=f"Altın Fiyat Tahmini — {sure_sec} ({pd.Timestamp.today().strftime('%d %B %Y')})",
    xaxis_title="Tarih",
    yaxis_title="$/ons",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ============================================
# GARCH VOLATİLİTE GRAFİĞİ
# ============================================
st.markdown("### 📉 GARCH Volatilite Analizi")
vol = garch.conditional_volatility

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=vol.index, y=vol.values,
    fill="tozeroy",
    fillcolor="rgba(148, 0, 211, 0.2)",
    line=dict(color="purple", width=1.5),
    name="Koşullu Volatilite"
))
fig2.update_layout(
    template="plotly_dark",
    height=300,
    title="GARCH(1,1) Koşullu Volatilite — 2015-Günümüz",
    xaxis_title="Tarih",
    yaxis_title="Volatilite (%)"
)
st.plotly_chart(fig2, use_container_width=True)

# ============================================
# MODEL DETAYLARI (GİZLENEBİLİR)
# ============================================
with st.expander("🔬 Model İstatistikleri (Detay)"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OLS Robust (HC3) Katsayılar**")
        katsayilar = pd.DataFrame({
            "Katsayı": ols.params.round(4),
            "p-değeri": ols.pvalues.round(4),
            "Anlamlı": ["✅" if p < 0.05 else "❌" for p in ols.pvalues]
        })
        st.dataframe(katsayilar)
    with col2:
        st.markdown("**GARCH(1,1) Parametreler**")
        garch_params = pd.DataFrame({
            "Parametre": ["omega", "alpha[1]", "beta[1]", "nu"],
            "Değer": [
                garch.params["omega"],
                garch.params["alpha[1]"],
                garch.params["beta[1]"],
                garch.params["nu"]
            ]
        }).set_index("Parametre").round(4)
        st.dataframe(garch_params)

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
<div style='text-align:center; color: gray; font-size: 12px'>
⚠️ Bu platform yatırım tavsiyesi değildir. Ekonometrik model çıktılarıdır.<br>
VAR + GARCH + OLS Robust | Veri: Yahoo Finance | © 2026
</div>
""", unsafe_allow_html=True)