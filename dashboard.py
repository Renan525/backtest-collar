import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from math import log, sqrt, exp, erf
import requests


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
    """Gera série de retornos do IBOV na mesma janela de datas do df (data_inicio/data_fim)."""
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
    Carrega Selic diária (série 11 do SGS) via API do Banco Central
    e calcula fator diário equivalente.
    """
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float) / 100.0  # % → decimal (anual)
    df = df.set_index("data").sort_index()
    # fator diário equivalente à taxa anual informada
    df["fator_diario"] = (1 + df["valor"]) ** (1 / DIAS_ANO) - 1
    return df


def riskfree_periodo(selic_df: pd.DataFrame, ini, fim) -> float:
    """
    Retorno acumulado da Selic no período [ini, fim].
    Se não tiver dados, retorna 0.
    """
    serie = selic_df.loc[ini:fim]["fator_diario"]
    if serie.empty:
        return 0.0
    return (1 + serie).prod() - 1


def obter_r_ano_selic(selic_df: pd.DataFrame, data) -> float:
    """
    Obtém a taxa anual (valor) da Selic para a data,
    usando o último valor disponível até esse dia.
    """
    serie = selic_df["valor"].loc[:data]
    if serie.empty:
        return selic_df["valor"].iloc[0]
    return serie.iloc[-1]


# ============================================================
# FUNÇÕES – COLLAR
# ============================================================

def backtest_collar(
    precos: pd.Series,
    dividendos: pd.Series,
    selic_df: pd.DataFrame,
    prazo_du: int,
    ganho_max: float,
    perda_max: float,
    dias_ano: int = DIAS_ANO,
):
    datas = precos.index

    if len(datas) <= prazo_du:
        return None

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    # Dividendos na janela
    ret_div = []
    riskfree_ops = []
    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]

        soma = 0.0
        if not dividendos.empty:
            soma = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma / p0[i])

        # Selic acumulada no período
        rf = riskfree_periodo(selic_df, ini, fim)
        riskfree_ops.append(rf)

    ret_div = np.array(ret_div)
    riskfree_ops = np.array(riskfree_ops)

    # Payoff da collar
    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0.0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0.0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho),
    )

    # Estrutura favorável: proteção atuou OU não houve limitação de ganho
    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)

    # Retorno com dividendos (para comparação com Selic do período)
    ret_op_com_div = ret_op_sem_div + ret_div

    # Agora a comparação é direta: retorno da operação vs Selic acumulada no período
    bate_cdi = (ret_op_com_div > riskfree_ops).astype(int)

    df = pd.DataFrame(
        {
            "data_inicio": datas[:-prazo_du],
            "data_fim": datas[prazo_du:],
            "ret_preco": ret_preco,
            "ret_dividendos": ret_div,
            "ret_op_sem_div": ret_op_sem_div,
            "ret_op_com_div": ret_op_com_div,
            "selic_periodo": riskfree_ops,
            "deu_certo": deu_certo,
            "bate_cdi": bate_cdi,
        }
    )

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_cdi": bate_cdi.mean(),
    }

    return df, resumo, dividendos


def gerar_grafico_collar(df: pd.DataFrame, ticker: str) -> BytesIO:
    df_plot = df.copy()
    df_plot["ret_ibov"] = gerar_ret_ibov(df_plot)

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_op_com_div"], label=f"Collar – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Retornos por operação – Collar x IBOV (não acumulado)", fontsize=14, weight="bold")
    plt.xlabel("Data de início da operação")
    plt.ylabel("Retorno no período")
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

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Preço justo de put europeia via Black-Scholes (σ anual, r anual)."""
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        return max(K - S0 * exp(-r * T), 0.0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    put_price = K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    return max(put_price, 0.0)


def estimar_vol_anual(precos: pd.Series, dias_ano: int = DIAS_ANO) -> float:
    """Vol anual histórica baseada em retornos logarítmicos."""
    if len(precos) < 2:
        return 1e-6
    log_ret = np.log(precos / precos.shift(1)).dropna()
    if log_ret.empty:
        return 1e-6
    sigma_diaria = log_ret.std()
    return max(sigma_diaria * np.sqrt(dias_ano), 1e-6)


def backtest_ap(
    precos: pd.Series,
    dividendos: pd.Series,
    selic_df: pd.DataFrame,
    prazo_du: int,
    perda_max: float,
    dias_ano: int = DIAS_ANO,
):
    """Backtest da AP com vol DINÂMICA por janela (até 252 dias antes da data de início)
       e comparação com Selic acumulada no período.
    """
    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    # Vol global como fallback
    sigma_global = estimar_vol_anual(precos, dias_ano=dias_ano)

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    preco_put_bsl = []
    custo_put_pct = []
    sigmas_usadas = []
    riskfree_ops = []

    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]

        # Dividendos na janela
        soma_div = 0.0
        if not dividendos.empty:
            soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma_div / p0[i])

        # Vol LOCAL: histórico até a data de início (últimos 252 dias)
        hist_pre = precos.loc[:ini].tail(dias_ano)
        sigma_local = estimar_vol_anual(hist_pre, dias_ano=dias_ano)
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigmas_usadas.append(sigma_local)

        # Risk-free anual (Selic) para precificar a put (último valor até ini)
        r_ano_local = obter_r_ano_selic(selic_df, ini)

        # Black-Scholes para essa janela
        S0 = p0[i]
        K = S0 * (1 - perda_max)
        T = prazo_du / dias_ano

        put_price = black_scholes_put(S0, K, r_ano_local, sigma_local, T)
        preco_put_bsl.append(put_price)
        custo_put_pct.append(put_price / S0)

        # Selic acumulada no período da operação (para comparação de retorno)
        rf = riskfree_periodo(selic_df, ini, fim)
        riskfree_ops.append(rf)

    ret_div = np.array(ret_div)
    preco_put_bsl = np.array(preco_put_bsl)
    custo_put_pct = np.array(custo_put_pct)
    sigmas_usadas = np.array(sigmas_usadas)
    riskfree_ops = np.array(riskfree_ops)

    # Retornos
    ret_ap_sem_div = ret_preco - custo_put_pct
    ret_ap_com_div = ret_preco + ret_div - custo_put_pct

    # Hedge acionado (preço final abaixo da perda máxima)
    hedge_acionado = (ret_preco <= -perda_max).astype(int)

    # Estrutura favorecida: hedge acionado OU retorno ≥ 0
    deu_certo = ((hedge_acionado == 1) | (ret_ap_com_div >= 0)).astype(int)

    # "Bate CDI" agora = retorno AP vs Selic acumulada no período
    bate_cdi = (ret_ap_com_div > riskfree_ops).astype(int)

    df = pd.DataFrame(
        {
            "data_inicio": datas[:-prazo_du],
            "data_fim": datas[prazo_du:],
            "preco_put_bsl": preco_put_bsl,
            "ret_preco": ret_preco,
            "ret_dividendos": ret_div,
            "custo_put_pct": custo_put_pct,
            "ret_ap_sem_div": ret_ap_sem_div,
            "ret_ap_com_div": ret_ap_com_div,
            "selic_periodo": riskfree_ops,
            "hedge_acionado": hedge_acionado,
            "deu_certo": deu_certo,
            "bate_cdi": bate_cdi,
            "sigma_local": sigmas_usadas,
        }
    )

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_cdi": bate_cdi.mean(),
    }

    return df, resumo, dividendos


def gerar_grafico_ap(df: pd.DataFrame, ticker: str) -> BytesIO:
    df_plot = df.copy()
    df_plot["ret_ibov"] = gerar_ret_ibov(df_plot)

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_ap_com_div"], label=f"AP – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black")
    plt.title("Retornos por operação – AP x IBOV (não acumulado)", fontsize=14, weight="bold")
    plt.xlabel("Data de início da operação")
    plt.ylabel("Retorno no período")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# STREAMLIT – DASHBOARD UNIFICADO
# ============================================================

st.set_page_config(page_title="Backtest Estruturas – Collar & AP", layout="wide")

st.title("📈 Backtest de Estruturas – Collar & Alocação Protegida")
st.markdown(
    """
    Ambiente unificado para análise de **Collar** e **Alocação Protegida (AP)**,
    com preços reais (Yahoo), dividendos, e **Selic histórica diária via Banco Central**.
    """
)

# Carregar Selic (uma vez, com cache)
try:
    selic_df = carregar_selic_com_fator()
except Exception as e:
    selic_df = None
    st.error("❌ Não foi possível carregar a Selic histórica do Banco Central. Verifique sua conexão com a internet.")


tab_collar, tab_ap = st.tabs(["📊 Collar", "🛡️ Alocação Protegida (AP)"])


# ------------------------------------------------------------
# ABA COLLAR
# ------------------------------------------------------------
with tab_collar:
    st.subheader("📊 Estratégia Collar")

    if selic_df is None:
        st.warning("Sem Selic histórica carregada, o backtest não pode rodar.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Configurações – Collar**")

        ticker_c = st.sidebar.text_input("Ticker (Collar):", "EZTC3.SA", key="ticker_c")

        prazo_du_c = st.sidebar.number_input(
            "Prazo da operação (dias úteis) – Collar:",
            value=63,
            step=1,
            format="%d",
            key="prazo_c",
        )

        ganho_max_c = st.sidebar.number_input(
            "Ganho máximo (%):",
            value=8.0,
            step=0.1,
            format="%.2f",
            key="ganho_c",
        ) / 100

        perda_max_c = st.sidebar.number_input(
            "Perda máxima (%):",
            value=8.0,
            step=0.1,
            format="%.2f",
            key="perda_c",
        ) / 100

        rodar_c = st.sidebar.button("🚀 Rodar Collar", key="rodar_collar")

        if rodar_c:
            precos_c, dividendos_c = carregar_preco_e_dividendos(ticker_c)
            resultado_c = backtest_collar(precos_c, dividendos_c, selic_df, prazo_du_c, ganho_max_c, perda_max_c)

            if resultado_c is None:
                st.error("Histórico insuficiente para esse prazo na aba Collar.")
            else:
                df_c, resumo_c, dividendos_c = resultado_c

                col1, col2 = st.columns(2)
                col1.metric("Estrutura Favorável (%)", f"{resumo_c['pct_deu_certo']*100:.1f}%")
                col2.metric("Bateu Selic (%)", f"{resumo_c['pct_bate_cdi']*100:.1f}%")

                st.subheader("📌 Dividendos (data EX – Yahoo) – Collar")
                if dividendos_c.empty:
                    st.warning("Nenhum dividendo encontrado no período.")
                else:
                    st.dataframe(dividendos_c.rename("valor_por_acao"))

                graf_c = gerar_grafico_collar(df_c, ticker_c)
                st.image(graf_c, caption="Retornos por operação – Collar x IBOV")

                st.subheader("📄 Detalhamento – Collar")
                st.dataframe(df_c)


# ------------------------------------------------------------
# ABA AP – ALOCAÇÃO PROTEGIDA
# ------------------------------------------------------------
with tab_ap:
    st.subheader("🛡️ Estratégia Alocação Protegida (AP)")

    if selic_df is None:
        st.warning("Sem Selic histórica carregada, o backtest não pode rodar.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Configurações – AP**")

        ticker_ap = st.sidebar.text_input("Ticker (AP):", "EZTC3.SA", key="ticker_ap")

        prazo_du_ap = st.sidebar.number_input(
            "Prazo da operação (dias úteis) – AP:",
            value=63,
            step=1,
            format="%d",
            key="prazo_ap",
        )

        perda_max_ap = st.sidebar.number_input(
            "Perda máxima protegida (%):",
            value=5.0,
            step=0.1,
            format="%.2f",
            key="perda_ap",
        ) / 100

        rodar_ap = st.sidebar.button("🚀 Rodar AP", key="rodar_ap")

        if rodar_ap:
            precos_ap, dividendos_ap = carregar_preco_e_dividendos(ticker_ap)
            resultado_ap = backtest_ap(precos_ap, dividendos_ap, selic_df, prazo_du_ap, perda_max_ap)

            if resultado_ap is None:
                st.error("Histórico insuficiente para esse prazo na aba AP.")
            else:
                df_ap, resumo_ap, dividendos_ap = resultado_ap

                # Preço justo ATUAL da put (HOJE) usando Selic & vol recentes
                sigma_atual = estimar_vol_anual(precos_ap.tail(DIAS_ANO))
                S0_atual = precos_ap.iloc[-1]
                data_atual = precos_ap.index[-1]
                r_ano_atual = obter_r_ano_selic(selic_df, data_atual)
                K_atual = S0_atual * (1 - perda_max_ap)
                T_atual = prazo_du_ap / DIAS_ANO

                preco_put_hoje = black_scholes_put(
                    S0=S0_atual,
                    K=K_atual,
                    r=r_ano_atual,
                    sigma=sigma_atual,
                    T=T_atual,
                )
                custo_pct_hoje = preco_put_hoje / S0_atual

                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Estrutura Favorável (%)", f"{resumo_ap['pct_deu_certo']*100:.1f}%")
                col2.metric("Bateu Selic (%)", f"{resumo_ap['pct_bate_cdi']*100:.1f}%")
                col3.metric("Preço Justo Atual da Put (R$)", f"R$ {preco_put_hoje:.4f}")
                col4.metric("Custo Atual da Put (% do ativo)", f"{custo_pct_hoje*100:.2f}%")

                # Dividendos
                st.subheader("📌 Dividendos (data EX – Yahoo) – AP")
                if dividendos_ap.empty:
                    st.warning("Nenhum dividendo encontrado no período.")
                else:
                    st.dataframe(dividendos_ap.rename("valor_por_acao"))

                # Preço justo da put e vol local por operação
                st.subheader("📘 Preço justo da Put (BSL), Selic do período e Vol local – histórico")
                st.dataframe(df_ap[["data_inicio", "data_fim", "preco_put_bsl", "custo_put_pct", "selic_periodo", "sigma_local"]])

                # Gráfico AP x IBOV
                graf_ap = gerar_grafico_ap(df_ap, ticker_ap)
                st.image(graf_ap, caption="Retornos por operação – AP x IBOV")

                # Detalhamento
                st.subheader("📄 Detalhamento – AP")
                st.dataframe(df_ap)
