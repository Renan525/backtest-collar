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
# PARÂMETROS GERAIS
# ============================================================

DIAS_ANO = 252


# ============================================================
# FUNÇÕES AUXILIARES – PREÇOS, DIVIDENDOS, IBOV
# ============================================================

def carregar_preco_e_dividendos(ticker: str):
    """Carrega preços (Close) e dividendos (data ex) do Yahoo."""
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


def gerar_ret_ibov(df_datas: pd.DataFrame):
    """Gera série de retornos do IBOV na mesma janela de datas do df."""
    ibov = yf.Ticker("^BVSP").history(period="2y", auto_adjust=False)["Close"]

    try:
        ibov.index = ibov.index.tz_localize(None)
    except Exception:
        pass

    ibov_ret = []
    for i in range(len(df_datas)):
        ini = df_datas.iloc[i]["data_inicio"]
        fim = df_datas.iloc[i]["data_fim"]
        if ini in ibov.index and fim in ibov.index:
            ibov_ret.append(ibov.loc[fim] / ibov.loc[ini] - 1)
        else:
            ibov_ret.append(np.nan)
    return np.array(ibov_ret)


# ============================================================
# FUNÇÕES – SELIC HISTÓRICA (BACEN)
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_selic_com_fator():
    """
    Carrega Selic diária via API do Banco Central (últimos 3 anos)
    e calcula fator diário equivalente.
    """
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
    df["valor"] = df["valor"].astype(float) / 100.0  # % → decimal anual
    df = df.set_index("data").sort_index()

    # fator diário equivalente
    df["fator_diario"] = (1 + df["valor"]) ** (1 / DIAS_ANO) - 1
    return df


def riskfree_periodo(selic_df: pd.DataFrame, ini, fim) -> float:
    """
    Retorno acumulado da Selic no período [ini, fim].
    """
    serie = selic_df.loc[ini:fim]["fator_diario"]
    if serie.empty:
        return 0.0
    return (1 + serie).prod() - 1


def obter_r_ano_selic(selic_df: pd.DataFrame, data) -> float:
    """
    Obtém a taxa anual (valor) da Selic para precificação da put.
    Pega o último valor disponível até a data.
    """
    serie = selic_df["valor"].loc[:data]
    return serie.iloc[-1] if not serie.empty else selic_df["valor"].iloc[0]


# ============================================================
# FUNÇÕES – BLACK-SCHOLES & VOL HISTÓRICA
# ============================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Preço justo de put europeia via Black-Scholes."""
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        return max(K - S0 * exp(-r * T), 0.0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    put_price = K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    return max(put_price, 0.0)


def estimar_vol_anual(precos: pd.Series) -> float:
    """Vol anual histórica baseada em retornos logarítmicos."""
    if len(precos) < 2:
        return 1e-6
    log_ret = np.log(precos / precos.shift(1)).dropna()
    if log_ret.empty:
        return 1e-6
    sigma_diaria = log_ret.std()
    return max(sigma_diaria * np.sqrt(DIAS_ANO), 1e-6)


# ============================================================
# FUNÇÕES – COLLAR
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
        ini = datas[i]
        fim = datas[i + prazo_du]

        # Dividendos
        soma_div = 0.0
        if not dividendos.empty:
            soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma_div / p0[i])

        # Selic acumulada do período
        selic_ops.append(riskfree_periodo(selic_df, ini, fim))

    ret_div = np.array(ret_div)
    selic_ops = np.array(selic_ops)

    # Payoff da collar
    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0.0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0.0)

    ret_op_sem_div = (
        np.where(
            (ret_defesa == 0) & (limit_ganho == 0),
            ret_preco,
            np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho),
        )
    )

    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)
    ret_op_com_div = ret_op_sem_div + ret_div

    bate_selic = (ret_op_com_div > selic_ops).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_op_sem_div": ret_op_sem_div,
        "ret_op_com_div": ret_op_com_div,
        "selic_periodo": selic_ops,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean(),
    }

    return df, resumo, dividendos


def gerar_grafico_collar(df, ticker):
    df_plot = df.copy()
    df_plot["ret_ibov"] = gerar_ret_ibov(df_plot)

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_op_com_div"], label=f"Collar – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black")
    plt.title("Retornos por operação – Collar x IBOV")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# FUNÇÕES – AP (ALOCAÇÃO PROTEGIDA)
# ============================================================

def backtest_ap(precos, dividendos, selic_df, prazo_du, perda_max):

    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    sigma_global = estimar_vol_anual(precos)

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    preco_put_bsl = []
    custo_put_pct = []
    sigmas_usadas = []
    selic_ops = []

    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]

        # Dividendos
        soma_div = 0.0
        if not dividendos.empty:
            soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma_div / p0[i])

        # Vol dinâmica
        hist_pre = precos.loc[:ini].tail(DIAS_ANO)
        sigma_local = estimar_vol_anual(hist_pre)
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigmas_usadas.append(sigma_local)

        # Selic anual para precificação da put
        r_ano_local = obter_r_ano_selic(selic_df, ini)

        # Black–Scholes
        S0 = p0[i]
        K = S0 * (1 - perda_max)
        T = prazo_du / DIAS_ANO

        preco_put = black_scholes_put(S0, K, r_ano_local, sigma_local, T)
        preco_put_bsl.append(preco_put)
        custo_put_pct.append(preco_put / S0)

        # Selic acumulada no período
        selic_ops.append(riskfree_periodo(selic_df, ini, fim))

    ret_div = np.array(ret_div)
    preco_put_bsl = np.array(preco_put_bsl)
    custo_put_pct = np.array(custo_put_pct)
    sigmas_usadas = np.array(sigmas_usadas)
    selic_ops = np.array(selic_ops)

    # Retornos da AP
    ret_ap_sem_div = ret_preco - custo_put_pct
    ret_ap_com_div = ret_preco + ret_div - custo_put_pct

    hedge_acionado = (ret_preco <= -perda_max).astype(int)
    deu_certo = ((hedge_acionado == 1) | (ret_ap_com_div >= 0)).astype(int)

    bate_selic = (ret_ap_com_div > selic_ops).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "preco_put_bsl": preco_put_bsl,
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "custo_put_pct": custo_put_pct,
        "ret_ap_sem_div": ret_ap_sem_div,
        "ret_ap_com_div": ret_ap_com_div,
        "selic_periodo": selic_ops,
        "hedge_acionado": hedge_acionado,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
        "sigma_local": sigmas_usadas,
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean(),
    }

    return df, resumo, dividendos


def gerar_grafico_ap(df, ticker):
    df_plot = df.copy()
    df_plot["ret_ibov"] = gerar_ret_ibov(df_plot)

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_ap_com_div"], label=f"AP – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black")
    plt.title("Retornos por operação – AP x IBOV")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# STREAMLIT – DASHBOARD COMPLETO
# ============================================================

st.set_page_config(page_title="Backtest – Collar & AP", layout="wide")

st.title("📈 Backtest de Estruturas – Collar & Alocação Protegida")
st.markdown(
    """
    Backtest completo usando **preços reais**, **dividendos**,  
    **volatilidade dinâmica**, e **Selic diária oficial do Banco Central**.
    """
)

# Carregar Selic
try:
    selic_df = carregar_selic_com_fator()
except Exception:
    st.error("❌ Erro ao carregar a Selic do Banco Central.")
    st.stop()

tab_collar, tab_ap = st.tabs(["📊 Collar", "🛡️ Alocação Protegida (AP)"])


# ------------------------------------------------------------
# COLLAR
# ------------------------------------------------------------
with tab_collar:
    st.subheader("📊 Estratégia Collar")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configurações – Collar**")

    ticker_c = st.sidebar.text_input("Ticker (Collar):", "EZTC3.SA", key="ticker_c")
    prazo_du_c = st.sidebar.number_input("Prazo (dias úteis) – Collar", 10, 252, 63, key="prazo_c")
    ganho_max_c = st.sidebar.number_input("Ganho Máx (%)", 0.0, 100.0, 8.0, key="ganho_c") / 100
    perda_max_c = st.sidebar.number_input("Perda Máx (%)", 0.0, 100.0, 8.0, key="perda_c") / 100

    rodar_c = st.sidebar.button("🚀 Rodar Collar", key="rodar_c")

    if rodar_c:
        precos_c, dividendos_c = carregar_preco_e_dividendos(ticker_c)
        resultado_c = backtest_collar(precos_c, dividendos_c, selic_df, prazo_du_c, ganho_max_c, perda_max_c)

        if resultado_c is None:
            st.error("Histórico insuficiente.")
        else:
            df_c, resumo_c, dividendos_c = resultado_c

            col1, col2 = st.columns(2)
            col1.metric("Estrutura Favorável (%)", f"{resumo_c['pct_deu_certo']*100:.1f}%")
            col2.metric("Bateu Selic (%)", f"{resumo_c['pct_bate_selic']*100:.1f}%")

            st.subheader("Dividendos")
            if dividendos_c.empty:
                st.info("Nenhum dividendo no período.")
            else:
                st.dataframe(dividendos_c.rename("valor_por_acao"))

            st.subheader("Gráfico")
            graf = gerar_grafico_collar(df_c, ticker_c)
            st.image(graf)

            st.subheader("Detalhamento")
            st.dataframe(df_c)


# ------------------------------------------------------------
# AP – ALOCAÇÃO PROTEGIDA
# ------------------------------------------------------------
with tab_ap:
    st.subheader("🛡️ Estratégia Alocação Protegida (AP)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configurações – AP**")

    ticker_ap = st.sidebar.text_input("Ticker (AP):", "EZTC3.SA", key="ticker_ap")
    prazo_du_ap = st.sidebar.number_input("Prazo (dias úteis) – AP", 10, 252, 63, key="prazo_ap")
    perda_max_ap = st.sidebar.number_input("Perda Máxima Protegida (%)", 0.0, 100.0, 5.0, key="perda_ap") / 100

    rodar_ap = st.sidebar.button("🚀 Rodar AP", key="rodar_ap")

    if rodar_ap:
        precos_ap, dividendos_ap = carregar_preco_e_dividendos(ticker_ap)
        resultado_ap = backtest_ap(precos_ap, dividendos_ap, selic_df, prazo_du_ap, perda_max_ap)

        if resultado_ap is None:
            st.error("Histórico insuficiente.")
        else:
            df_ap, resumo_ap, dividendos_ap = resultado_ap

            # Cálculo do preço da PUT hoje
            sigma_atual = estimar_vol_anual(precos_ap.tail(DIAS_ANO))
            S0_atual = precos_ap.iloc[-1]
            K_atual = S0_atual * (1 - perda_max_ap)
            data_atual = precos_ap.index[-1]
            r_ano_atual = obter_r_ano_selic(selic_df, data_atual)
            T_atual = prazo_du_ap / DIAS_ANO

            preco_put_hoje = black_scholes_put(S0_atual, K_atual, r_ano_atual, sigma_atual, T_atual)
            custo_pct_hoje = preco_put_hoje / S0_atual

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Estrutura Favorável (%)", f"{resumo_ap['pct_deu_certo']*100:.1f}%")
            col2.metric("Bateu Selic (%)", f"{resumo_ap['pct_bate_selic']*100:.1f}%")
            col3.metric("PUT justa hoje (R$)", f"R$ {preco_put_hoje:.4f}")
            col4.metric("Custo PUT (% ativo)", f"{custo_pct_hoje*100:.2f}%")

            st.subheader("Dividendos")
            if dividendos_ap.empty:
                st.info("Nenhum dividendo no período.")
            else:
                st.dataframe(dividendos_ap.rename("valor_por_acao"))

            st.subheader("Histórico: PUT, Selic e Vol")
            st.dataframe(df_ap[["data_inicio", "data_fim", "preco_put_bsl", "custo_put_pct", "selic_periodo", "sigma_local"]])

            st.subheader("Gráfico")
            graf = gerar_grafico_ap(df_ap, ticker_ap)
            st.image(graf)

            st.subheader("Detalhamento")
            st.dataframe(df_ap)
