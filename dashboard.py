import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from math import log, sqrt, exp, erf
import requests
import datetime

# ============================================================
# PARÂMETROS
# ============================================================

DIAS_ANO = 252


# ============================================================
# FUNÇÕES AUXILIARES – PREÇOS E DIVIDENDOS
# ============================================================

def carregar_preco_e_dividendos(ticker: str):
    ativo = yf.Ticker(ticker)

    hist = ativo.history(period="2y", auto_adjust=False)
    precos = hist["Close"].dropna()

    try:
        precos.index = precos.index.tz_localize(None)
    except:
        pass

    dividendos = ativo.dividends
    try:
        dividendos.index = dividendos.index.tz_localize(None)
    except:
        pass

    return precos, dividendos


def gerar_ret_ibov(df_datas: pd.DataFrame):
    ibov = yf.Ticker("^BVSP").history(period="2y", auto_adjust=False)["Close"]

    try:
        ibov.index = ibov.index.tz_localize(None)
    except:
        pass

    ret = []
    for i in range(len(df_datas)):
        ini = df_datas.iloc[i]["data_inicio"]
        fim = df_datas.iloc[i]["data_fim"]

        if ini in ibov.index and fim in ibov.index:
            ret.append(ibov.loc[fim] / ibov.loc[ini] - 1)
        else:
            ret.append(np.nan)

    return np.array(ret)


# ============================================================
# SELIC HISTÓRICA – Banco Central
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_selic_com_fator():
    hoje = datetime.date.today()
    data_final = hoje.strftime("%d/%m/%Y")
    data_inicial = (hoje - datetime.timedelta(days=3*365)).strftime("%d/%m/%Y")

    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/"
        f"dados?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float) / 100.0
    df = df.set_index("data").sort_index()

    df["fator_diario"] = (1 + df["valor"]) ** (1 / DIAS_ANO) - 1
    return df


def riskfree_periodo(selic_df: pd.DataFrame, ini, fim):
    idx = pd.date_range(start=ini, end=fim, freq="D")
    serie = selic_df["fator_diario"].reindex(idx).ffill()
    return (1 + serie).prod() - 1, serie


def obter_r_ano_selic(selic_df: pd.DataFrame, data) -> float:
    serie = selic_df["valor"].loc[:data]
    return serie.iloc[-1] if not serie.empty else selic_df["valor"].iloc[0]


# ============================================================
# BLACK-SCHOLES + VOL HISTÓRICA
# ============================================================

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def estimar_vol_anual(precos: pd.Series):
    if len(precos) < 2:
        return 1e-6
    log_ret = np.log(precos / precos.shift(1)).dropna()
    if log_ret.empty:
        return 1e-6
    return log_ret.std() * np.sqrt(252)


def black_scholes_put(S0, K, r, sigma, T):
    if T <= 0:
        return max(K - S0, 0)
    if sigma <= 0:
        return max(K - S0 * exp(-r * T), 0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return max(K * exp(-r*T) * norm_cdf(-d2) - S0 * norm_cdf(-d1), 0)


# ============================================================
# COLLAR
# ============================================================

def backtest_collar(precos, dividendos, selic_df, prazo_du, ganho_max, perda_max):

    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    selic_periodos = []
    selic_debug_list = []

    for i in range(len(p0)):
        ini, fim = datas[i], datas[i + prazo_du]

        soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum() if not dividendos.empty else 0
        ret_div.append(soma_div / p0[i])

        selic_acum, lista = riskfree_periodo(selic_df, ini, fim)
        selic_periodos.append(selic_acum)
        selic_debug_list.append(lista)

    ret_div = np.array(ret_div)
    selic_periodos = np.array(selic_periodos)

    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho)
    )

    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)
    ret_op_com_div = ret_op_sem_div + ret_div
    bate_selic = (ret_op_com_div > selic_periodos).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_op_sem_div": ret_op_sem_div,
        "ret_op_com_div": ret_op_com_div,
        "selic_periodo": selic_periodos,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
        "selic_detalhada": selic_debug_list
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean()
    }

    return df, resumo, dividendos


# ============================================================
# AP – Alocação Protegida (COM SPREAD)
# ============================================================

def backtest_ap(precos, dividendos, selic_df, prazo_du, perda_max, preco_put_cotada):

    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    # -----------------------------
    # 1. Calcular preço justo da PUT HOJE (BSL)
    # -----------------------------
    sigma_atual = estimar_vol_anual(precos.tail(252))
    S0_atual = precos.iloc[-1]
    K_atual = S0_atual * (1 - perda_max)
    T_atual = prazo_du / 252
    r_atual = obter_r_ano_selic(selic_df, datas[-1])

    preco_put_justo_hoje = black_scholes_put(S0_atual, K_atual, r_atual, sigma_atual, T_atual)

    # -----------------------------
    # 2. Spread entre PUT cotada e justa
    # -----------------------------
    if preco_put_justo_hoje <= 0:
        markup = 1.0    # fallback
    else:
        markup = preco_put_cotada / preco_put_justo_hoje

    # -----------------------------
    # 3. Histórico
    # -----------------------------
    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div, selic_periodos, selic_debug_list = [], [], []
    preco_put_bsl_hist, preco_put_ajustado_hist = [], []
    custo_put_pct = []
    sigma_local_hist = []

    sigma_global = estimar_vol_anual(precos)

    for i in range(len(p0)):
        ini, fim = datas[i], datas[i + prazo_du]

        soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum() if not dividendos.empty else 0
        ret_div.append(soma_div / p0[i])

        # Vol local
        hist_pre = precos.loc[:ini].tail(252)
        sigma_local = estimar_vol_anual(hist_pre)
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigma_local_hist.append(sigma_local)

        # Selic período
        selic_acum, lista = riskfree_periodo(selic_df, ini, fim)
        selic_periodos.append(selic_acum)
        selic_debug_list.append(lista)

        # Preço justo histórico
        S0 = p0[i]
        K = S0 * (1 - perda_max)
        T = prazo_du / 252
        r_local = obter_r_ano_selic(selic_df, ini)

        preco_put_hist_justo = black_scholes_put(S0, K, r_local, sigma_local, T)
        preco_put_hist_ajustado = preco_put_hist_justo * markup

        preco_put_bsl_hist.append(preco_put_hist_justo)
        preco_put_ajustado_hist.append(preco_put_hist_ajustado)
        custo_put_pct.append(preco_put_hist_ajustado / S0)

    ret_div = np.array(ret_div)
    custo_put_pct = np.array(custo_put_pct)
    selic_periodos = np.array(selic_periodos)
    sigma_local_hist = np.array(sigma_local_hist)

    # -----------------------------
    # 4. Retornos finais
    # -----------------------------
    ret_ap_sem_div = ret_preco - custo_put_pct
    ret_ap_com_div = ret_preco + ret_div - custo_put_pct

    hedge_acionado = (ret_preco <= -perda_max).astype(int)
    deu_certo = ((hedge_acionado == 1) | (ret_ap_com_div >= 0)).astype(int)
    bate_selic = (ret_ap_com_div > selic_periodos).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "preco_put_justo_hist": preco_put_bsl_hist,
        "preco_put_ajustado_hist": preco_put_ajustado_hist,
        "markup_aplicado": markup,
        "custo_put_pct": custo_put_pct,
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_ap_sem_div": ret_ap_sem_div,
        "ret_ap_com_div": ret_ap_com_div,
        "selic_periodo": selic_periodos,
        "hedge_acionado": hedge_acionado,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
        "sigma_local": sigma_local_hist,
        "selic_detalhada": selic_debug_list
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean(),
        "preco_put_justo_hoje": preco_put_justo_hoje,
        "preco_put_cotado_hoje": preco_put_cotada,
        "markup": markup
    }

    return df, resumo, dividendos


# ============================================================
# GRÁFICOS
# ============================================================

def gerar_grafico_collar(df, ticker):
    dfp = df.copy()
    dfp["ret_ibov"] = gerar_ret_ibov(dfp)

    plt.figure(figsize=(12, 5))
    plt.plot(dfp["data_inicio"], dfp["ret_op_com_div"], label=f"Collar – {ticker}")
    plt.plot(dfp["data_inicio"], dfp["ret_ibov"], label="IBOV")
    plt.axhline(0, color="black")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


def gerar_grafico_ap(df, ticker):
    dfp = df.copy()
    dfp["ret_ibov"] = gerar_ret_ibov(dfp)

    plt.figure(figsize=(12, 5))
    plt.plot(dfp["data_inicio"], dfp["ret_ap_com_div"], label=f"AP – {ticker}")
    plt.plot(dfp["data_inicio"], dfp["ret_ibov"], label="IBOV")
    plt.axhline(0, color="black")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# DASHBOARD
# ============================================================

st.set_page_config(page_title="Backtest – Estruturas", layout="wide")
st.title("📈 Backtest – Collar & AP (com Spread da PUT)")

st.markdown("Inclui Selic real acumulada, volatilidade dinâmica e spread ajustado nas puts históricas.")

try:
    selic_df = carregar_selic_com_fator()
except:
    st.error("Erro ao carregar Selic.")
    st.stop()


tab_c, tab_ap = st.tabs(["📊 Collar", "🛡️ AP (com spread)"])


# COLLAR
with tab_c:
    st.subheader("📊 Collar")

    ticker_c = st.text_input("Ticker:", "EZTC3.SA", key="tick_c")
    prazo_c = st.number_input("Prazo (dias úteis)", 10, 252, 63, key="prazo_c")
    ganho_c = st.number_input("Ganho Máx (%)", 0.0, 30.0, 8.0, key="g_c") / 100
    perda_c = st.number_input("Perda Máx (%)", 0.0, 30.0, 8.0, key="p_c") / 100

    if st.button("Rodar Collar"):
        precos, divs = carregar_preco_e_dividendos(ticker_c)
        resultado = backtest_collar(precos, divs, selic_df, prazo_c, ganho_c, perda_c)

        if resultado:
            df, res, _ = resultado

            c1, c2 = st.columns(2)
            c1.metric("Estrutura Favorável", f"{res['pct_deu_certo']*100:.1f}%")
            c2.metric("Bateu Selic", f"{res['pct_bate_selic']*100:.1f}%")

            st.image(gerar_grafico_collar(df, ticker_c))
            st.dataframe(df)


# AP COM SPREAD
with tab_ap:
    st.subheader("🛡️ Alocação Protegida (com spread aplicado)")

    ticker_ap = st.text_input("Ticker:", "EZTC3.SA", key="tick_ap")
    prazo_ap = st.number_input("Prazo (dias úteis)", 10, 252, 63, key="prazo_ap")
    perda_ap = st.number_input("Perda Máxima Protegida (%)", 0.0, 30.0, 5.0, key="p_ap") / 100
    preco_put_cotada = st.number_input("Preço cotado da PUT hoje (R$):", 0.01, 50.0, 0.50, key="put_input")

    if st.button("Rodar AP"):
        precos, divs = carregar_preco_e_dividendos(ticker_ap)
        resultado = backtest_ap(precos, divs, selic_df, prazo_ap, perda_ap, preco_put_cotada)

        if resultado:
            df, res, _ = resultado

            c1, c2, c3 = st.columns(3)
            c1.metric("PUT justa (BSL)", f"R$ {res['preco_put_justo_hoje']:.4f}")
            c2.metric("PUT cotada", f"R$ {res['preco_put_cotado_hoje']:.4f}")
            c3.metric("Spread aplicado", f"{(res['markup']-1)*100:.1f}%")

            c4, c5 = st.columns(2)
            c4.metric("Estrutura Favorável", f"{res['pct_deu_certo']*100:.1f}%")
            c5.metric("Bateu Selic", f"{res['pct_bate_selic']*100:.1f}%")

            st.image(gerar_grafico_ap(df, ticker_ap))

            st.subheader("Detalhamento")
            st.dataframe(df)
