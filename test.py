import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from financial_analyzer_enhanced_corrected import (
    obter_dados_fundamentalistas_detalhados_br,
    calcular_piotroski_f_score_br,
    calcular_value_composite_score,
    otimizar_portfolio_markowitz_mc,
    sugerir_alocacao_novo_aporte
)

st.title("Backtest Mensal com Aportes, Markowitz MC e Rebalanceamento Long Only (max 30% por ativo)")

# Configurações
valor_aporte = 1000.0
limite_porc_ativo = 0.3  # 30%
start_date = pd.to_datetime("2024-05-05")
end_date = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))

# Input
tickers_str = st.text_input("Tickers elegíveis (ex: PETR4.SA,VALE3.SA,ITUB4.SA)", "PETR4.SA,VALE3.SA,ITUB4.SA")
tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

if st.button("Executar Backtest Mensal"):
    if not tickers:
        st.warning("Inclua ao menos um ticker.")
        st.stop()
    # Preparação
    datas_aporte = pd.date_range(start_date, end_date, freq="MS")
    valor_carteira = []
    datas_carteira = []
    historico_pesos = []
    historico_num_ativos = []
    patrimonio = 0.0

    # Histórico dos preços
    st.write("Baixando preços históricos de todos ativos elegíveis e do benchmark (BOVA11.SA)...")
    all_tickers = list(set(tickers + ['BOVA11.SA']))
    precos = yf.download(all_tickers, start=start_date, end=end_date)
    # Seleção robusta da coluna de preço
    if isinstance(precos.columns, pd.MultiIndex):
        if 'Adj Close' in precos.columns.get_level_values(0):
            precos = precos.xs('Adj Close', axis=1, level=0)
        elif 'Close' in precos.columns.get_level_values(0):
            precos = precos.xs('Close', axis=1, level=0)
        else:
            st.error("Colunas de preço não encontradas!")
            st.stop()
    else:
        if 'Adj Close' in precos.columns:
            precos = precos[['Adj Close']]
        elif 'Close' in precos.columns:
            precos = precos[['Close']]
        else:
            st.error("Colunas de preço não encontradas!")
            st.stop()
    precos = precos.dropna(how='all', axis=0)

    # Inicialização
    carteira = {t: 0 for t in tickers}
    caixa = 0.0

    for idx, data_aporte in enumerate(datas_aporte):
        st.write(f"Processando mês: {data_aporte.strftime('%Y-%m')}")
        # Período para cálculo dos retornos e fundamentos
        data_fim_mes = data_aporte + relativedelta(months=1) - timedelta(days=1)
        data_fim_mes = min(data_fim_mes, end_date)
        period_prices = precos.loc[:data_fim_mes].copy()

        # Recalcula fundamentos e scores
        df_fund = obter_dados_fundamentalistas_detalhados_br(tickers)
        if df_fund.empty:
            st.warning(f"Sem dados fundamentalistas para {data_aporte.strftime('%Y-%m')}. Pulando mês.")
            valor_carteira.append(patrimonio)
            datas_carteira.append(data_aporte)
            continue
        df_fund['Piotroski_F_Score'] = df_fund.apply(lambda row: float(calcular_piotroski_f_score_br(row)), axis=1)
        vc_metrics = {
            'trailingPE': 'lower_is_better', 'priceToBook': 'lower_is_better', 
            'enterpriseToEbitda': 'lower_is_better', 'dividendYield': 'higher_is_better',
            'returnOnEquity': 'higher_is_better', 'netMargin': 'higher_is_better'
        }
        # Garante que retorna uma Series numérica
        quant_value_score = calcular_value_composite_score(df_fund, vc_metrics)
        if not isinstance(quant_value_score, pd.Series):
            quant_value_score = pd.Series(quant_value_score, index=df_fund.index)
        df_fund['Quant_Value_Score'] = quant_value_score.apply(lambda x: float(x) if np.isscalar(x) else np.nan)

        # Seleciona ativos
        selecionados = df_fund[(df_fund['Piotroski_F_Score'] >= 5) & (df_fund['Quant_Value_Score'] >= 0.5)]
        ativos_validos = [t for t in selecionados['ticker'].tolist() if t in period_prices.columns and period_prices[t].notna().any()]
        if not ativos_validos:
            st.warning(f"Nenhum ativo passou no filtro em {data_aporte.strftime('%Y-%m')}. Pulando mês.")
            valor_carteira.append(patrimonio)
            datas_carteira.append(data_aporte)
            continue

        # Calcula retornos históricos para otimização (usa últimos 12 meses)
        lookback_inicio = data_aporte - relativedelta(months=120)
        lookback_prices = precos.loc[lookback_inicio:data_aporte, ativos_validos].dropna()
        if len(lookback_prices) < 2:
            st.warning(f"Dados insuficientes para otimização em {data_aporte.strftime('%Y-%m')}. Pulando mês.")
            valor_carteira.append(patrimonio)
            datas_carteira.append(data_aporte)
            continue

        returns = lookback_prices.pct_change().dropna()

        # Otimização (Markowitz MC)
        portfolio, _ = otimizar_portfolio_markowitz_mc(
            ativos_validos, returns, taxa_livre_risco=0.14
        )
        # Limita a 30% por ativo e normaliza
        pesos = pd.Series(portfolio['pesos'])
        pesos = pesos.clip(upper=limite_porc_ativo)
        pesos = pesos / pesos.sum()
        portfolio['pesos'] = pesos.to_dict()

        # Sugerir alocação do novo aporte (função do seu módulo)
        if data_aporte in period_prices.index:
            precos_mes = period_prices.loc[data_aporte, ativos_validos]
        else:
            precos_mes = period_prices.loc[period_prices.index.asof(data_aporte), ativos_validos]
        valores_ativos_atuais = {ativo: carteira.get(ativo, 0) * precos_mes[ativo] for ativo in ativos_validos}
        aportes, _ = sugerir_alocacao_novo_aporte(
            current_portfolio_composition_values=valores_ativos_atuais,
            new_capital=valor_aporte,
            target_portfolio_weights_decimal=portfolio['pesos']
        )

        # Atualiza quantidades da carteira (compras a mercado)
        for ativo in ativos_validos:
            valor_compra = aportes.get(ativo, 0)
            if valor_compra > 0 and precos_mes[ativo] > 0:
                qtd = int(valor_compra // precos_mes[ativo])
                carteira[ativo] = carteira.get(ativo, 0) + qtd

        # Atualiza patrimônio com preços do fim do mês (último pregão disponível)
        data_ultima = period_prices.index.asof(data_fim_mes)
        precos_fim = period_prices.loc[data_ultima, ativos_validos]
        patrimonio = sum(carteira.get(ativo, 0) * precos_fim[ativo] for ativo in ativos_validos)
        valor_carteira.append(patrimonio)
        datas_carteira.append(data_ultima)
        historico_pesos.append(portfolio['pesos'])
        historico_num_ativos.append(len(ativos_validos))

    # Simular aportes mensais no BOVA11 (benchmark DCA)
    bova11_prices = precos['BOVA11.SA']
    bova11_quantidade = 0
    bova11_patrimonio = []

    for dt in datas_carteira:
        preco = bova11_prices.asof(dt)
        if np.isnan(preco):
            bova11_patrimonio.append(np.nan)
            continue
        qtd_comprada = valor_aporte // preco
        bova11_quantidade += qtd_comprada
        patrimonio_bova = bova11_quantidade * preco
        bova11_patrimonio.append(patrimonio_bova)

    df_result = pd.DataFrame({
        'Carteira Quant': valor_carteira,
        'BOVA11': bova11_patrimonio
    }, index=datas_carteira)

    st.line_chart(df_result)

    st.write("Evolução da carteira e benchmark")
    st.write(df_result)

    # Métricas finais
    n_years = (df_result.index[-1] - df_result.index[0]).days / 365.25
    total_aportado = valor_aporte * len(datas_aporte)
    carteira_cagr = (df_result['Carteira Quant'].iloc[-1] / total_aportado) ** (1/n_years) - 1
    bova_cagr = (df_result['BOVA11'].iloc[-1] / total_aportado) ** (1/n_years) - 1
    st.metric("CAGR Carteira Quant", f"{carteira_cagr:.2%}")
    st.metric("CAGR BOVA11", f"{bova_cagr:.2%}")

    st.write("Número de ativos por mês:", historico_num_ativos)
    st.write("Pesos por mês:", historico_pesos)
    st.success("Backtest mensal com aportes e rebalanceamento concluído!")

st.caption("Backtest mensal, rebalanceando e aportando usando funções Markowitz MC e alocação do repositório Jaovm/QUANT. Limite de 30% por ativo, long only, Piotroski e Quant Score recalculados mensalmente.")
