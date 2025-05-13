import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go

# Importando funções do seu módulo do repositório
from financial_analyzer_enhanced_corrected import (
    obter_dados_fundamentalistas_detalhados_br,
    calcular_piotroski_f_score_br,
    calcular_value_composite_score
)

st.title('Backtest - Piotroski + Quant Value Score (Long Only) vs BOVA11')

st.markdown("""
Este painel faz o filtro automático dos ativos usando os scores do modelo do repositório (Piotroski ≥ 6 e Quant Value Score ≥ 6),
busca os dados via yfinance e compara o resultado com o BOVA11 a partir de 01-01-2024.
""")

# Input do usuário
tickers_str = st.text_input(
    "Tickers separados por vírgula (ex: PETR4.SA,VALE3.SA,ITUB4.SA)", 
    "PETR4.SA,VALE3.SA,ITUB4.SA"
)
tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

if st.button("Executar Backtest"):
    if not tickers:
        st.warning("Inclua pelo menos um ticker.")
        st.stop()
        
    st.write("Obtendo dados fundamentalistas...")
    df_fund = obter_dados_fundamentalistas_detalhados_br(tickers)
    if df_fund.empty:
        st.error("Não foi possível obter dados fundamentalistas.")
        st.stop()
    
    # Calcula scores
    st.write("Calculando scores...")
    df_fund['Piotroski_F_Score'] = df_fund.apply(lambda row: calcular_piotroski_f_score_br(row), axis=1)
    vc_metrics = {
        'trailingPE': 'lower_is_better', 'priceToBook': 'lower_is_better', 
        'enterpriseToEbitda': 'lower_is_better', 'dividendYield': 'higher_is_better',
        'returnOnEquity': 'higher_is_better', 'netMargin': 'higher_is_better'
    }
    df_fund['Quant_Value_Score'] = calcular_value_composite_score(df_fund, vc_metrics)
    
    # Filtro
    st.write("Filtrando ativos (Piotroski ≥ 6 e QuantValue ≥ 6)...")
    selecionados = df_fund[(df_fund['Piotroski_F_Score'] >= 6) & (df_fund['Quant_Value_Score'] >= 6)]
    st.dataframe(selecionados[['ticker', 'Piotroski_F_Score', 'Quant_Value_Score']])
    ativos_validos = selecionados['ticker'].tolist()
    
    if not ativos_validos:
        st.warning("Nenhum ativo passou nos critérios.")
        st.stop()
    
    # Preços históricos
    st.write("Baixando preços históricos...")
    start_date = "2024-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    tickers_yf = ativos_validos + ['BOVA11.SA']
    data = yf.download(tickers_yf, start=start_date, end=end_date)
    st.write("Colunas retornadas:", data.columns)  # debug
    
    # Corrige para MultiIndex ou coluna simples
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(1):
            data = data.xs('Adj Close', axis=1, level=1)
        else:
            st.error(f"'Adj Close' não encontrado nas colunas! Colunas retornadas: {data.columns}")
            st.stop()
    else:
        if 'Adj Close' in data.columns:
            data = data[['Adj Close']]
        else:
            st.error(f"'Adj Close' não encontrado nas colunas! Colunas retornadas: {data.columns}")
            st.stop()

    data = data.dropna()
    if len(data) < 2:
        st.error("Dados insuficientes para backtest.")
        st.stop()
        
    # Calculando evolução do portfólio (equal weight, rebalanceamento diário)
    portf = data[ativos_validos].pct_change().dropna().mean(axis=1)
    portf_cum = (1 + portf).cumprod()
    bova = data['BOVA11.SA'].pct_change().dropna()
    bova_cum = (1 + bova).cumprod()
    
    # Gráfico
    st.subheader("Evolução do Portfólio vs BOVA11")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portf_cum.index, y=portf_cum, name='Portfólio Quant'))
    fig.add_trace(go.Scatter(x=bova_cum.index, y=bova_cum, name='BOVA11'))
    fig.update_layout(title="Crescimento do capital (base 1.0)", yaxis_title="Evolução", xaxis_title="Data")
    st.plotly_chart(fig)
    
    # CAGR
    n_years = (portf_cum.index[-1] - portf_cum.index[0]).days / 365.25
    portf_cagr = portf_cum[-1] ** (1/n_years) - 1
    bova_cagr = bova_cum[-1] ** (1/n_years) - 1
    st.metric("CAGR Portfólio", f"{portf_cagr:.2%}")
    st.metric("CAGR BOVA11", f"{bova_cagr:.2%}")
    st.success("Backtest concluído! Veja o gráfico e as métricas acima.")

st.caption("Código baseado no modelo e funções do repositório Jaovm/QUANT.")
