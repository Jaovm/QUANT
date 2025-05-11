import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Painel Avançado de Otimização de Carteira QUANTOVITZ")

# --- Importar funções do financial_analyzer_enhanced_corrected.py ---
try:
    from financial_analyzer_enhanced_corrected import (
        obter_dados_historicos_yf,
        obter_dados_fundamentalistas_detalhados_br,
        calcular_piotroski_f_score_br,
        calcular_value_composite_score,
        get_fama_french_factors,
        calcular_beneish_m_score,
        calcular_altman_z_score,
        otimizar_portfolio_scipy,
        otimizar_portfolio_markowitz_mc,
        calcular_metricas_portfolio, 
        sugerir_alocacao_novo_aporte,
        RISK_FREE_RATE_DEFAULT
    )
    st.sidebar.success("Módulo de análise corrigido carregado!")
except ImportError as e:
    st.error(f"Erro ao importar o módulo 'financial_analyzer_enhanced_corrected.py': {e}. Certifique-se de que o arquivo está no diretório correto e todas as dependências (arch, statsmodels) estão instaladas.")
    st.stop()

# --- Entradas do Usuário na Sidebar ---
st.sidebar.header("Parâmetros da Análise")

# 1. Carteira Atual
st.sidebar.subheader("1. Carteira Atual")
ativos_input_str = st.sidebar.text_input("Ativos da carteira (ex: PETR4.SA,VALE3.SA,ITUB4.SA)", "PETR4.SA,VALE3.SA,ITUB4.SA")
pesos_input_str = st.sidebar.text_input("Pesos percentuais da carteira (ex: 40,30,30)", "40,30,30")
valor_total_carteira_atual = st.sidebar.number_input("Valor total da carteira atual (R$)", min_value=0.0, value=100000.0, step=1000.0)

# 2. Novo Aporte (Opcional)
st.sidebar.subheader("2. Novo Aporte (Opcional)")
novo_capital_input = st.sidebar.number_input("Novo capital a ser aportado (R$)", min_value=0.0, value=10000.0, step=100.0)

# 3. Ativos Candidatos (Opcional)
st.sidebar.subheader("3. Ativos Candidatos (Adicionar à Análise)")
candidatos_input_str = st.sidebar.text_input("Ativos candidatos (ex: MGLU3.SA,WEGE3.SA)", "MGLU3.SA,WEGE3.SA")

# 4. Período de Análise de Dados Históricos
st.sidebar.subheader("4. Período de Análise Histórica")
start_date_analise_input = st.sidebar.date_input("Data Inicial para Dados Históricos", datetime.today() - timedelta(days=5*365))
end_date_analise_input = st.sidebar.date_input("Data Final para Dados Históricos", datetime.today())
start_date_analise = start_date_analise_input.strftime("%Y-%m-%d")
end_date_analise = end_date_analise_input.strftime("%Y-%m-%d")

# 5. Taxa Livre de Risco
st.sidebar.subheader("5. Taxa Livre de Risco (Anual)")
taxa_livre_risco_input = st.sidebar.number_input("Taxa Livre de Risco (ex: 0.02 para 2%)", min_value=0.0, max_value=1.0, value=RISK_FREE_RATE_DEFAULT, step=0.001, format="%.3f")

# 6. Configurações de Otimização Avançada
st.sidebar.subheader("6. Configurações de Otimização Avançada")
vc_metrics_selection = st.sidebar.multiselect(
    "Métricas para Value Composite Score (VC2/VC6)",
    options=[
        'trailingPE', 'priceToBook', 'enterpriseToEbitda', 
        'dividendYield', 'returnOnEquity', 'netMargin', 
        'forwardPE', 'marketCap'
    ],
    default=[
        'trailingPE', 'priceToBook', 'enterpriseToEbitda', 
        'dividendYield', 'returnOnEquity', 'netMargin'
    ]
)
VC_METRIC_DIRECTIONS = {
    'trailingPE': 'lower_is_better', 
    'priceToBook': 'lower_is_better', 
    'enterpriseToEbitda': 'lower_is_better',
    'dividendYield': 'higher_is_better', 
    'returnOnEquity': 'higher_is_better',
    'netMargin': 'higher_is_better',
    'forwardPE': 'lower_is_better',
    'marketCap': 'lower_is_better'
}
vc_metrics_config = {metric: VC_METRIC_DIRECTIONS[metric] for metric in vc_metrics_selection if metric in VC_METRIC_DIRECTIONS}

min_piotroski_score = st.sidebar.slider(
    "Piotroski F-Score Mínimo para Inclusão de Ativos",
    0, 9, 0,
    help="Ativos com score abaixo deste valor podem ser excluídos da otimização avançada. 0 para não filtrar."
)

mostrar_detalhes_piotroski = st.sidebar.checkbox("Mostrar detalhes dos critérios do Piotroski F-Score")
            
# 7. Restrições de Peso na Otimização
st.sidebar.subheader("7. Restrições de Alocação (Otimização)")
min_aloc_global = st.sidebar.slider(
    "Alocação Mínima Global por Ativo (%)", 0, 40, 0, 
    help="Restrição inferior para cada ativo na carteira otimizada."
) / 100.0
max_aloc_global = st.sidebar.slider(
    "Alocação Máxima Global por Ativo (%)", 10, 100, 100, 
    help="Restrição superior para cada ativo na carteira otimizada."
) / 100.0

manter_pesos_atuais_opcao = st.sidebar.selectbox(
    "Considerar Pesos Atuais na Otimização Avançada",
    options=["Não considerar", "Como ponto de partida", "Como restrição inferior aproximada", "Como restrição de intervalo"],
    index=1, help="Define como os pesos da carteira atual informada são usados na otimização avançada."
)
tolerancia_peso_atual = st.sidebar.slider(
    "Tolerância para Restrição de Peso Atual (% do peso atual)",
    0, 100, 20,
    help="Usado se 'Como restrição de intervalo' for selecionado. Ex: 20% -> peso_atual*0.8 a peso_atual*1.2."
)

# --- Filtro Quant Value para novo aporte (integração com Markowitz) ---
st.sidebar.subheader("Filtro Quant Value para Novo Aporte")
top_n_quant_value = st.sidebar.number_input(
    "Comprar apenas os Top N ativos em Quant Value Score (0 = todos)", 
    min_value=0, max_value=20, value=0
)
min_quant_value = st.sidebar.slider(
    "Nota mínima Quant Value Score para compra (0 = sem filtro)", 
    min_value=0.0, max_value=1.0, value=0.0, step=0.01
)

run_analysis = st.sidebar.button("Executar Análise Avançada")

def key_to_str(k):
    if isinstance(k, tuple):
        return k[0]
    return str(k)
    
def plot_efficient_frontier_comparative(fronteiras_data, portfolios_otimizados, carteira_atual_metricas=None):
    if not any(f["pontos"] for f in fronteiras_data):
        st.write("Não foi possível gerar dados para a Fronteira Eficiente.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    idx_color = 0

    for i, data in enumerate(fronteiras_data):
        nome_fronteira = data["nome"]
        pontos = data["pontos"]
        if pontos:
            df_fronteira = pd.DataFrame(pontos)
            fig.add_trace(go.Scatter(x=df_fronteira['volatilidade']*100, y=df_fronteira['retorno']*100,
                                     mode='markers', name=f'Simulações ({nome_fronteira})',
                                     marker=dict(color=df_fronteira['sharpe'], colorscale='Viridis', showscale=(i==0), size=5, line=dict(width=0),
                                                 colorbar=dict(title='Sharpe Ratio')),
                                     text=[f"Sharpe: {s:.2f}<br>Pesos: {str({k: f'{v*100:.1f}%' for k,v in p.items()})[:100]}..." 
                                           for s, p in zip(df_fronteira['sharpe'], df_fronteira['pesos'])]))
    if carteira_atual_metricas:
        fig.add_trace(go.Scatter(x=[carteira_atual_metricas['volatilidade']*100], y=[carteira_atual_metricas['retorno_esperado']*100],
                                 mode='markers+text', name='Carteira Atual',
                                 marker=dict(color='black', size=12, symbol='diamond-tall'),
                                 text="Atual", textposition="top center",
                                 hovertext=f"Carteira Atual<br>Sharpe: {carteira_atual_metricas['sharpe_ratio']:.2f}"))

    portfolio_symbols = ['star', 'circle', 'cross', 'triangle-up', 'pentagon']
    for i, portfolio_info in enumerate(portfolios_otimizados):
        nome = portfolio_info['nome']
        portfolio = portfolio_info['data']
        if portfolio:
            fig.add_trace(go.Scatter(x=[portfolio['volatilidade']*100], y=[portfolio['retorno_esperado']*100],
                                     mode='markers+text', name=nome,
                                     marker=dict(color=colors[idx_color % len(colors)], size=14, symbol=portfolio_symbols[i % len(portfolio_symbols)]),
                                     text=nome.split(" ")[0],
                                     textposition="bottom right",
                                     hovertext=f"{nome}<br>Sharpe: {portfolio.get('sharpe_ratio', 0):.2f}<br>Pesos: {str({k: f'{v*100:.1f}%' for k,v in portfolio['Pesos'].items()})}"))
            idx_color +=1

    fig.update_layout(title='Fronteiras Eficientes e Carteiras Otimizadas',
                      xaxis_title='Volatilidade Anualizada (%)',
                      yaxis_title='Retorno Esperado Anualizado (%)',
                      legend_title_text='Legenda',
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_pie_chart(weights_dict, title):
    if not weights_dict or sum(weights_dict.values()) == 0:
        return
    df_pie = pd.DataFrame(list(weights_dict.items()), columns=['Ativo', 'Peso'])
    df_pie = df_pie[df_pie['Peso'] > 1e-4] # Filter out very small weights for cleaner pie chart
    fig = px.pie(df_pie, values='Peso', names='Ativo', title=title, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_comparative_table(carteiras_data):
    if not carteiras_data:
        st.write("Não há dados de carteiras para comparar.")
        return
    df_comparativo = pd.DataFrame(carteiras_data)
    df_comparativo_display = df_comparativo[['Nome', 'Retorno Esperado (%)', 'Volatilidade (%)', 'Sharpe Ratio']].copy()
    df_comparativo_display = df_comparativo_display.set_index('Nome')
    st.subheader("Tabela Comparativa de Desempenho")
    st.dataframe(df_comparativo_display.style.format("{:.2f}"))
    st.subheader("Composição Detalhada das Carteiras (%)")
    todos_ativos_pesos = set()
    for c_data in carteiras_data:
        c = c_data.get("Dados")
        if c and 'Pesos' in c and isinstance(c['Pesos'], dict):
            todos_ativos_pesos.update(c['Pesos'].keys())
    pesos_data_list = []
    for c_data in carteiras_data:
        c = c_data.get("Dados")
        row = {'Nome': c_data['Nome']}
        if c and 'Pesos' in c and isinstance(c['Pesos'], dict):
            for ativo in todos_ativos_pesos:
                row[ativo] = c['Pesos'].get(ativo, 0) * 100
        else:
            for ativo in todos_ativos_pesos:
                row[ativo] = 0
        pesos_data_list.append(row)
    df_pesos_detalhados = pd.DataFrame(pesos_data_list).set_index('Nome')
    st.dataframe(df_pesos_detalhados.style.format("{:.2f}"))
    
if run_analysis:
    st.header("Resultados da Análise Avançada")
    ativos_carteira_lista_raw = [s.strip().upper() for s in ativos_input_str.split(',') if s.strip()]
    try:
        pesos_carteira_lista_pct_raw = [float(p.strip()) for p in pesos_input_str.split(',') if p.strip()]
        if len(ativos_carteira_lista_raw) != len(pesos_carteira_lista_pct_raw):
            st.error("O número de ativos e pesos na carteira atual deve ser o mesmo.")
            st.stop()
        if not np.isclose(sum(pesos_carteira_lista_pct_raw), 100.0, atol=0.1):
            st.warning(f"A soma dos pesos da carteira atual ({sum(pesos_carteira_lista_pct_raw):.2f}%) não é 100%. Ajuste ou os resultados podem ser inconsistentes.")
        ativos_carteira_lista = []
        pesos_carteira_lista_pct = []
        for ativo, peso_pct in zip(ativos_carteira_lista_raw, pesos_carteira_lista_pct_raw):
            if peso_pct > 1e-4:
                ativos_carteira_lista.append(ativo)
                pesos_carteira_lista_pct.append(peso_pct)
        pesos_carteira_decimal = {ativo: peso/100.0 for ativo, peso in zip(ativos_carteira_lista, pesos_carteira_lista_pct)}
        carteira_atual_composicao_valores = {ativo: pesos_carteira_decimal[ativo] * valor_total_carteira_atual for ativo in ativos_carteira_lista}
    except ValueError:
        st.error("Os pesos da carteira atual devem ser números.")
        st.stop()
    ativos_candidatos_lista = [s.strip().upper() for s in candidatos_input_str.split(',') if s.strip()]
    todos_ativos_analise = sorted(list(set(ativos_carteira_lista + ativos_candidatos_lista)))
    if not todos_ativos_analise:
        st.error("Nenhum ativo fornecido para análise (carteira atual ou candidatos).")
        st.stop()
    st.info(f"Ativos para análise: {', '.join(todos_ativos_analise)}\nPeríodo histórico: {start_date_analise} a {end_date_analise}")
    df_retornos_historicos = pd.DataFrame()
    df_fundamental_completo = pd.DataFrame()
    fama_french_factors_df = pd.DataFrame()
    with st.spinner("Coletando e processando dados... Por favor, aguarde."):
        df_retornos_historicos = obter_dados_historicos_yf(todos_ativos_analise, start_date_analise, end_date_analise)
        if df_retornos_historicos.empty or df_retornos_historicos.shape[0] < 60:
            st.error(f"Não foi possível obter dados históricos suficientes para {', '.join(todos_ativos_analise)} no período especificado. Verifique os tickers e o período.")
            st.stop()
        df_fundamental_completo = obter_dados_fundamentalistas_detalhados_br(todos_ativos_analise)
        if not df_fundamental_completo.empty:
            df_fundamental_completo.set_index('ticker', inplace=True, drop=False)
            # Ajuste para a função calcular_piotroski_f_score_br que retorna (score, criterios, debug_valores) quando verbose=True
            piotroski_results = df_fundamental_completo.apply(lambda row: calcular_piotroski_f_score_br(row, verbose=True), axis=1)
            df_fundamental_completo["Piotroski_F_Score"] = piotroski_results.apply(lambda x: x[0] if isinstance(x, tuple) else x) # Pega o score
            df_fundamental_completo["Piotroski_F_Detalhes"] = piotroski_results.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None) # Pega os critérios
            # Se desejar os valores de debug, adicione uma nova coluna:
            # df_fundamental_completo["Piotroski_F_Debug"] = piotroski_results.apply(lambda x: x[2] if isinstance(x, tuple) and len(x) > 2 else None)
            
            df_fundamental_completo['Quant_Value_Score'] = calcular_value_composite_score(df_fundamental_completo, vc_metrics_config)
            df_fundamental_completo['Altman_Z_Score'] = df_fundamental_completo.apply(calcular_altman_z_score, axis=1)
            df_fundamental_completo['Beneish_M_Score'] = df_fundamental_completo.apply(calcular_beneish_m_score, axis=1)
            st.subheader("Dados Fundamentalistas e Scores")
            colunas_desejadas = [
                'ticker', 'Piotroski_F_Score', 'Quant_Value_Score',
                'Altman_Z_Score', 'Beneish_M_Score', 'enterpriseToEbitda', 'netMargin'
            ]
            # Adicionar colunas que podem ou não estar presentes devido à coleta de dados
            colunas_fund_existentes = [c for c in ['trailingPE', 'priceToBook', 'dividendYield', 'returnOnEquity'] if c in df_fundamental_completo.columns]
            colunas_desejadas.extend(colunas_fund_existentes)
            colunas_presentes = [c for c in colunas_desejadas if c in df_fundamental_completo.columns]
            st.dataframe(df_fundamental_completo[list(dict.fromkeys(colunas_presentes))]) # Remove duplicatas mantendo a ordem
            if mostrar_detalhes_piotroski and "Piotroski_F_Detalhes" in df_fundamental_completo.columns:
                try:
                    detalhes_df = df_fundamental_completo['Piotroski_F_Detalhes'].apply(pd.Series)
                    detalhes_df['ticker'] = df_fundamental_completo['ticker'].values # Adiciona o ticker para referência
                    st.dataframe(detalhes_df.set_index('ticker'))
                except Exception as e_details:
                    st.warning(f"Não foi possível exibir detalhes do Piotroski: {e_details}")
        else:
            st.warning("Não foi possível obter dados fundamentalistas. A otimização avançada pode ser limitada.")

        ff_start_date = (pd.to_datetime(start_date_analise) - timedelta(days=30)).strftime("%Y-%m-%d")
        fama_french_factors_df = get_fama_french_factors(ff_start_date, end_date_analise)
        if fama_french_factors_df.empty:
            st.warning("Não foi possível obter dados para os fatores Fama-French. Estimativas de Alpha/Beta não serão realizadas.")

    # --- Otimização e Análise de Carteiras ---
    st.header("Otimização de Carteira e Análise de Risco")
    carteiras_comparativo_lista = []
    fronteiras_plot_data = []
    portfolios_otimizados_plot_data = []
    carteira_atual_metricas_plot = None

    # Métricas da Carteira Atual
    if ativos_carteira_lista and not df_retornos_historicos.empty and len(ativos_carteira_lista) > 0:
        retornos_carteira_atual_df = df_retornos_historicos[ativos_carteira_lista]
        # Garantir que todos os ativos da carteira atual têm dados de retorno
        if retornos_carteira_atual_df.shape[1] == len(ativos_carteira_lista) and not retornos_carteira_atual_df.isnull().values.any():
            pesos_np_atuais = np.array([pesos_carteira_decimal[ativo] for ativo in ativos_carteira_lista])
            ret_med_atuais = retornos_carteira_atual_df.mean() * 252
            mat_cov_atuais = retornos_carteira_atual_df.cov() * 252
            
            if not ret_med_atuais.empty and not mat_cov_atuais.empty and len(pesos_np_atuais) == len(ret_med_atuais):
                ret_atual, vol_atual, sharpe_atual = calcular_metricas_portfolio(pesos_np_atuais, ret_med_atuais, mat_cov_atuais, taxa_livre_risco_input)
                carteira_atual_metricas_plot = {
                    'retorno_esperado': ret_atual,
                    'volatilidade': vol_atual,
                    'sharpe_ratio': sharpe_atual,
                    'Pesos': pesos_carteira_decimal
                }
                carteiras_comparativo_lista.append({
                    "Nome": "Carteira Atual",
                    "Retorno Esperado (%)": ret_atual * 100,
                    "Volatilidade (%)": vol_atual * 100,
                    "Sharpe Ratio": sharpe_atual,
                    "Dados": carteira_atual_metricas_plot
                })
                col_atual1, col_atual2 = st.columns(2)
                with col_atual1:
                    st.subheader("Desempenho da Carteira Atual")
                    st.metric("Retorno Anualizado", f"{ret_atual*100:.2f}%")
                    st.metric("Volatilidade Anualizada", f"{vol_atual*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe_atual:.2f}")
                with col_atual2:
                    plot_portfolio_pie_chart(pesos_carteira_decimal, "Composição da Carteira Atual")
            else:
                st.warning("Não foi possível calcular as métricas da carteira atual devido a dados insuficientes ou inconsistentes para os ativos especificados.")
        else:
            st.warning("Alguns ativos da carteira atual não possuem dados de retorno suficientes no período selecionado.")

    # Ativos para otimização (considerando filtros)
    ativos_para_otimizacao = todos_ativos_analise
    if not df_fundamental_completo.empty and min_piotroski_score > 0:
        ativos_filtrados_piotroski = df_fundamental_completo[df_fundamental_completo["Piotroski_F_Score"] >= min_piotroski_score].index.tolist()
        ativos_para_otimizacao = [a for a in ativos_para_otimizacao if a in ativos_filtrados_piotroski]
        st.write(f"Ativos após filtro Piotroski (>= {min_piotroski_score}): {', '.join(ativos_para_otimizacao) if ativos_para_otimizacao else 'Nenhum'}")

    if not ativos_para_otimizacao:
        st.warning("Nenhum ativo selecionado para otimização após filtros. Verifique os critérios.")
    elif df_retornos_historicos.empty or df_retornos_historicos[ativos_para_otimizacao].shape[1] < len(ativos_para_otimizacao) or df_retornos_historicos[ativos_para_otimizacao].isnull().values.any().any():
        st.warning(f"Dados históricos insuficientes ou com NaNs para um ou mais ativos selecionados para otimização: {', '.join(ativos_para_otimizacao)}. Otimização não será executada.")
    else:
        retornos_otimizacao = df_retornos_historicos[ativos_para_otimizacao]
        ret_med_otim = retornos_otimizacao.mean() * 252
        mat_cov_otim = retornos_otimizacao.cov() * 252

        # Fronteira Eficiente (Markowitz MC)
        with st.spinner("Calculando Fronteira Eficiente (Monte Carlo)... "):
            fronteira_mc_pontos, _, _ = otimizar_portfolio_markowitz_mc(ret_med_otim, mat_cov_otim, taxa_livre_risco_input, num_portfolios=5000)
            if fronteira_mc_pontos:
                fronteiras_plot_data.append({"nome": "Markowitz MC", "pontos": fronteira_mc_pontos})
        
        # Otimização por Sharpe Máximo (SciPy)
        with st.spinner("Otimizando para Máximo Sharpe Ratio (SciPy)... "):
            pesos_max_sharpe, ret_max_sharpe, vol_max_sharpe, sharpe_max = otimizar_portfolio_scipy(ret_med_otim, mat_cov_otim, taxa_livre_risco_input, objetivo="max_sharpe", min_retorno=None, bounds=(min_aloc_global, max_aloc_global))
            if pesos_max_sharpe is not None:
                portfolio_max_sharpe_data = {
                    'retorno_esperado': ret_max_sharpe,
                    'volatilidade': vol_max_sharpe,
                    'sharpe_ratio': sharpe_max,
                    'Pesos': dict(zip(ativos_para_otimizacao, pesos_max_sharpe))
                }
                portfolios_otimizados_plot_data.append({"nome": "Max Sharpe (SciPy)", "data": portfolio_max_sharpe_data})
                carteiras_comparativo_lista.append({
                    "Nome": "Max Sharpe (SciPy)",
                    "Retorno Esperado (%)": ret_max_sharpe * 100,
                    "Volatilidade (%)": vol_max_sharpe * 100,
                    "Sharpe Ratio": sharpe_max,
                    "Dados": portfolio_max_sharpe_data
                })

        # Otimização por Mínima Volatilidade (SciPy)
        with st.spinner("Otimizando para Mínima Volatilidade (SciPy)... "):
            pesos_min_vol, ret_min_vol, vol_min_vol, sharpe_min_vol = otimizar_portfolio_scipy(ret_med_otim, mat_cov_otim, taxa_livre_risco_input, objetivo="min_volatility", min_retorno=None, bounds=(min_aloc_global, max_aloc_global))
            if pesos_min_vol is not None:
                portfolio_min_vol_data = {
                    'retorno_esperado': ret_min_vol,
                    'volatilidade': vol_min_vol,
                    'sharpe_ratio': sharpe_min_vol,
                    'Pesos': dict(zip(ativos_para_otimizacao, pesos_min_vol))
                }
                portfolios_otimizados_plot_data.append({"nome": "Min Volatility (SciPy)", "data": portfolio_min_vol_data})
                carteiras_comparativo_lista.append({
                    "Nome": "Min Volatility (SciPy)",
                    "Retorno Esperado (%)": ret_min_vol * 100,
                    "Volatilidade (%)": vol_min_vol * 100,
                    "Sharpe Ratio": sharpe_min_vol,
                    "Dados": portfolio_min_vol_data
                })
        
        # Plotar Fronteira e Carteiras Otimizadas
        plot_efficient_frontier_comparative(fronteiras_plot_data, portfolios_otimizados_plot_data, carteira_atual_metricas_plot)

        # Exibir Gráficos de Pizza e Tabela Comparativa
        if portfolios_otimizados_plot_data:
            st.subheader("Composição das Carteiras Otimizadas")
            num_cols_pie = min(len(portfolios_otimizados_plot_data), 3) # Max 3 pie charts per row
            cols_pie = st.columns(num_cols_pie)
            for i, p_info in enumerate(portfolios_otimizados_plot_data):
                plot_portfolio_pie_chart(p_info['data']['Pesos'], p_info['nome'], col=cols_pie[i % num_cols_pie])

    # Tabela Comparativa Final
    if carteiras_comparativo_lista:
        display_comparative_table(carteiras_comparativo_lista)
    else:
        st.warning("Nenhuma carteira pôde ser otimizada ou analisada com os dados e configurações fornecidas.")

    # Sugestão de Alocação para Novo Aporte
    if novo_capital_input > 0 and portfolios_otimizados_plot_data:
        st.header("Sugestão de Alocação para Novo Aporte")
        carteira_referencia_nome = st.selectbox(
            "Usar qual carteira otimizada como referência para o novo aporte?",
            options=[p['nome'] for p in portfolios_otimizados_plot_data],
            index=0 # Default to Max Sharpe
        )
        carteira_referencia_data = next((p['data'] for p in portfolios_otimizados_plot_data if p['nome'] == carteira_referencia_nome), None)

        if carteira_referencia_data:
            pesos_referencia = carteira_referencia_data['Pesos']
            df_sugestao_aporte = sugerir_alocacao_novo_aporte(
                carteira_atual_composicao_valores if ativos_carteira_lista else {},
                valor_total_carteira_atual if ativos_carteira_lista else 0,
                novo_capital_input,
                pesos_referencia,
                df_fundamental=df_fundamental_completo if not df_fundamental_completo.empty else None,
                top_n_quant=top_n_quant_value,
                min_quant_score=min_quant_value
            )
            if not df_sugestao_aporte.empty:
                st.subheader(f"Alocação do Novo Aporte (R$ {novo_capital_input:,.2f}) com base em '{carteira_referencia_nome}'")
                st.dataframe(df_sugestao_aporte.style.format({
                    'Valor Atual': 'R$ {:,.2f}', 
                    'Peso Atual (%)': '{:.2f}%',
                    'Peso Ideal (%)': '{:.2f}%',
                    'Valor Ideal Após Aporte': 'R$ {:,.2f}',
                    'Comprar (R$)': 'R$ {:,.2f}',
                    'Novo Peso (%)': '{:.2f}%',
                    'Quant_Value_Score': '{:.3f}'
                }))
                
                # Gráfico de pizza para a carteira final após o aporte
                carteira_final_composicao = {}
                for idx, row_aporte in df_sugestao_aporte.iterrows():
                    carteira_final_composicao[row_aporte['Ativo']] = row_aporte['Valor Ideal Após Aporte']
                
                # Normalizar para pesos percentuais para o gráfico de pizza
                total_carteira_final = sum(carteira_final_composicao.values())
                if total_carteira_final > 0:
                    pesos_carteira_final_pct = {k: v / total_carteira_final for k, v in carteira_final_composicao.items()}
                    plot_portfolio_pie_chart(pesos_carteira_final_pct, f"Composição da Carteira Final Estimada (após aporte em '{carteira_referencia_nome}')")
            else:
                st.warning("Não foi possível gerar sugestão de alocação para o novo aporte com os critérios definidos.")
        else:
            st.error("Carteira de referência para o novo aporte não encontrada.")
    elif novo_capital_input > 0:
        st.warning("Novo aporte informado, mas nenhuma carteira otimizada está disponível para servir de referência.")

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Análise Avançada' para ver os resultados.")


