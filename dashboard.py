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
# FUNÇÕES AUXILIARES – PREÇOS, DIVIDENDOS, IBOV
# ============================================================

def carregar_preco_e_dividendos(ticker: str):
    ativo = yf.Ticker(ticker)
    hist = ativo.history(period="2y", auto_adjust=False)
    precos = hist["Close"].dropna()

    try:
        precos.index = precos.index.tz_localize(None)
    except Exception:
        pass

    dividendos = ativo.dividends
    try:
        dividendos.index = dividendos.index.tz_localize(None)
    except Exception:
        pass

    return precos, dividendos


def gerar_ret_ibov(df_datas):
    ibov = yf.Ticker("^BVSP").history(period="2y", auto_adjust=False)["Close"]
    try:
        ibov.index = ibov.index.tz_localize(None)
    except Exception:
        pass

    ibov_ret = []
    for i in range(len(df_datas)):
        ini, fim = df_datas.iloc[i]["data_inicio"], df_datas.iloc[i]["data_fim"]
        if ini in ibov.index and fim in ibov.index:
            ibov_ret.append(ibov.loc[fim] / ibov.loc[ini] - 1)
        else:
            ibov_ret.append(np.nan)
    return np.array(ibov_ret)


# ============================================================
# SELIC – BANCO CENTRAL
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
    df["valor"] = df["valor"].astype(float) / 100
    df = df.set_index("data").sort_index()

    df["fator_diario"] = (1 + df["valor"]) ** (1 / DIAS_ANO) - 1
    return df


def riskfree_periodo(selic_df, ini, fim):
    serie = selic_df.loc[ini:fim]["fator_diario"]
    if serie.empty:
        return 0
    return (1 + serie).prod() - 1


def obter_r_ano_selic(selic_df, data):
    serie = selic_df["valor"].loc[:data]
    return serie.iloc[-1] if not serie.empty else selic_df["valor"].iloc[0]


# ============================================================
# BLACK–SCHOLES E VOL
# ============================================================

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes_put(S0, K, r, sigma, T):
    if T <= 0:
        return max(K - S0, 0)
    if sigma <= 0:
        return max(K - S0 * exp(-r * T), 0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return max(K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1), 0)


def estimar_vol_anual(precos):
    if len(precos) < 2:
        return 1e-6
    log_ret = np.log(precos / precos.shift(1)).dropna()
    if log_ret.empty:
        return 1e-6
    return max(log_ret.std() * np.sqrt(DIAS_ANO), 1e-6)


# ============================================================
# BACKTEST COLLAR
# ============================================================

def backtest_collar(precos, dividendos, selic_df, prazo_du, ganho_max, perda_max):

    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div, selic_ops = [], []

    for i in range(len(p0)):
        ini, fim = datas[i], datas[i + prazo_du]

        soma = 0
        if not dividendos.empty:
            soma = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma / p0[i])

        selic_ops.append(riskfree_periodo(selic_df, ini, fim))

    ret_div = np.array(ret_div)
    selic_ops = np.array(selic_ops)

    # Payoff
    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho),
    )

    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)
    ret_op_com_div = ret_op_sem_div + ret_div
    bate_selic = (ret_op_com_div > selic_ops).astype(int)

    # Selic anual equivalente por operação
    selic_anual = (1 + selic_ops) ** (DIAS_ANO / prazo_du) - 1

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_op_com_div": ret_op_com_div,
        "ret_ibov": gerar_ret_ibov(pd.DataFrame({"data_inicio": datas[:-prazo_du], "data_fim": datas[prazo_du:]})),
        "selic_anual": selic_anual,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean()
    }

    return df, resumo, dividendos


def gerar_grafico_collar(df, ticker):
    plt.figure(figsize=(12, 5))
    plt.plot(df["data_inicio"], df["ret_op_com_div"], label=f"Collar – {ticker}", linewidth=2)
    plt.plot(df["data_inicio"], df["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.plot(df["data_inicio"], df["selic_anual"], label="Selic Anual Equivalente", linewidth=2, linestyle="--", color="green")

    plt.axhline(0, color="black")
    plt.title("Retornos – Collar x IBOV x Selic Anual", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# BACKTEST AP (ALOCACÃO PROTEGIDA)
# ============================================================

def backtest_ap(precos, dividendos, selic_df, prazo_du, perda_max):

    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    sigma_global = estimar_vol_anual(precos)

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div, preco_put, custo_pct, sigma_local_list, selic_ops = [], [], [], [], []

    for i in range(len(p0)):
        ini, fim = datas[i], datas[i + prazo_du]

        # Dividendos
        soma = 0
        if not dividendos.empty:
            soma = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma / p0[i])

        # Vol dinâmica
        sigma_local = estimar_vol_anual(precos.loc[:ini].tail(DIAS_ANO))
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigma_local_list.append(sigma_local)

        # Selic
        r_ano = obter_r_ano_selic(selic_df, ini)

        # PUT justa
        S0, K = p0[i], p0[i] * (1 - perda_max)
        T = prazo_du / DIAS_ANO
        preco = black_scholes_put(S0, K, r_ano, sigma_local, T)
        preco_put.append(preco)
        custo_pct.append(preco / S0)

        selic_ops.append(riskfree_periodo(selic_df, ini, fim))

    ret_div = np.array(ret_div)
    preco_put = np.array(preco_put)
    custo_pct = np.array(custo_pct)
    sigma_local_list = np.array(sigma_local_list)
    selic_ops = np.array(selic_ops)

    ret_ap_com_div = ret_preco + ret_div - custo_pct

    hedge_acionado = (ret_preco <= -perda_max).astype(int)
    deu_certo = ((hedge_acionado == 1) | (ret_ap_com_div >= 0)).astype(int)
    bate_selic = (ret_ap_com_div > selic_ops).astype(int)

    selic_anual = (1 + selic_ops) ** (DIAS_ANO / prazo_du) - 1

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_ap_com_div": ret_ap_com_div,
        "ret_ibov": gerar_ret_ibov(pd.DataFrame({"data_inicio": datas[:-prazo_du], "data_fim": datas[prazo_du:]})),
        "selic_anual": selic_anual,
        "preco_put_bsl": preco_put,
        "custo_put_pct": custo_pct,
        "sigma_local": sigma_local_list,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean()
    }

    return df, resumo, dividendos


def gerar_grafico_ap(df, ticker):
    plt.figure(figsize=(12, 5))
    plt.plot(df["data_inicio"], df["ret_ap_com_div"], label=f"AP – {ticker}", linewidth=2)
    plt.plot(df["data_inicio"], df["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.plot(df["data_inicio"], df["selic_anual"], label="Selic Anual Equivalente", linewidth=2, linestyle="--", color="green")

    plt.axhline(0, color="black")
    plt.title("Retornos – AP x IBOV x Selic Anual", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# DASHBOARD STREAMLIT
# ============================================================

st.set_page_config(page_title="Backtest – Estruturas", layout="wide")

st.title("📈 Backtest de Estruturas – Collar e Alocação Protegida")

st.markdown("""
 Ferramenta institucional com:
- Preços reais (Yahoo)
- Dividendos
- Volatilidade dinâmica
- Selic diária (BACEN)
- Comparação contra IBOV e Selic Anual Equivalente
""")

# Carregar Selic
try:
    selic_df = carregar_selic_com_fator()
except:
    st.error("Erro ao carregar Selic.")
    st.stop()

tab_c, tab_ap = st.tabs(["📊 Collar", "🛡️ Alocação Protegida"])


# ============================
# COLLAR
# ============================

with tab_c:
    st.sidebar.subheader("Configurações – Collar")

    ticker_c = st.sidebar.text_input("Ticker – Collar", "EZTC3.SA", key="ticker_c")
    prazo_du_c = st.sidebar.number_input("Prazo (DU)", 10, 252, 63, key="prazo_c")
    ganho_max_c = st.sidebar.number_input("Ganho Máx (%)", 0.0, 50.0, 8.0, key="ganho_c") / 100
    perda_max_c = st.sidebar.number_input("Perda Máx (%)", 0.0, 50.0, 8.0, key="perda_c") / 100

    if st.sidebar.button("📊 Rodar Collar", key="rodar_c"):
        precos_c, dividends_c = carregar_preco_e_dividendos(ticker_c)
        res = backtest_collar(precos_c, dividends_c, selic_df, prazo_du_c, ganho_max_c, perda_max_c)

        if res is None:
            st.error("Histórico insuficiente!")
        else:
            df, resumo, dividends = res

            col1, col2 = st.columns(2)
            col1.metric("Estrutura Favorável", f"{resumo['pct_deu_certo']*100:.1f}%")
            col2.metric("Bateu Selic", f"{resumo['pct_bate_selic']*100:.1f}%")

            st.subheader("Gráfico – Collar x IBOV x Selic")
            st.image(gerar_grafico_collar(df, ticker_c))

            st.subheader("Detalhamento")
            st.dataframe(df)


# ============================
# AP
# ============================

with tab_ap:
    st.sidebar.subheader("Configurações – AP")

    ticker_ap = st.sidebar.text_input("Ticker – AP", "EZTC3.SA", key="ticker_ap")
    prazo_du_ap = st.sidebar.number_input("Prazo (DU)", 10, 252, 63, key="prazo_ap")
    perda_max_ap = st.sidebar.number_input("Perda Máx Protegida (%)", 0.0, 50.0, 5.0, key="perda_ap") / 100

    if st.sidebar.button("🛡️ Rodar AP", key="rodar_ap"):
        precos_ap, dividends_ap = carregar_preco_e_dividendos(ticker_ap)
        res = backtest_ap(precos_ap, dividends_ap, selic_df, prazo_du_ap, perda_max_ap)

        if res is None:
            st.error("Histórico insuficiente!")
        else:
            df, resumo, dividends = res

            col1, col2 = st.columns(2)
            col1.metric("Estrutura Favorável", f"{resumo['pct_deu_certo']*100:.1f}%")
            col2.metric("Bateu Selic", f"{resumo['pct_bate_selic']*100:.1f}%")

            st.subheader("Gráfico – AP x IBOV x Selic")
            st.image(gerar_grafico_ap(df, ticker_ap))

            st.subheader("Detalhamento")
            st.dataframe(df)
