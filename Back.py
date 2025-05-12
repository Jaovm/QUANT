# backtest_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import plotly.graph_objects as go
import os
import sys

# Adicionar o diretório do script do analisador ao path
# Supondo que o script de backtest será salvo em /home/ubuntu/quant_project/
# e o financial_analyzer_enhanced_corrected.py está em /home/ubuntu/quant_project/QUANT-main/
sys.path.append("/home/ubuntu/quant_project/QUANT-main")

# Importar funções do script fornecido
try:
    from financial_analyzer_enhanced_corrected import (
        obter_dados_fundamentalistas_detalhados_br,
        calcular_piotroski_f_score_br,
        _get_numeric_value # Helper function usada em calcular_piotroski_f_score_br
        # A função otimizar_portfolio_scipy será definida abaixo caso não seja importável ou para garantir a assinatura correta.
    )
except ImportError as e:
    # Se financial_analyzer_enhanced_corrected não puder ser importado, 
    # funções cruciais como obter_dados_fundamentalistas_detalhados_br e calcular_piotroski_f_score_br faltarão.
    # O script não poderá funcionar. Vamos definir stubs ou parar.
    st.error(f"Erro crítico ao importar financial_analyzer_enhanced_corrected.py: {e}. Funções essenciais podem estar faltando.")
    # Para permitir que o restante do script seja definido, vamos criar stubs se a importação falhar.
    def obter_dados_fundamentalistas_detalhados_br(ativos):
        st.warning("STUB: obter_dados_fundamentalistas_detalhados_br não importada.")
        return pd.DataFrame(index=ativos, columns=['lucro_liquido_atual', 'cfo_atual', 'ativos_totais_atual', 'ativos_totais_anterior', 'lucro_liquido_anterior', 'divida_lp_atual', 'divida_lp_anterior', 'ativos_circulantes_atual', 'passivos_circulantes_atual', 'ativos_circulantes_anterior', 'passivos_circulantes_anterior', 'receita_liquida_atual', 'receita_liquida_anterior', 'lucro_bruto_atual', 'lucro_bruto_anterior', 'sharesOutstanding_from_bs_curr', 'sharesOutstanding_from_bs_prev'])

    def calcular_piotroski_f_score_br(row, verbose=False):
        st.warning("STUB: calcular_piotroski_f_score_br não importada.")
        return 0 # Retorna um score baixo para não selecionar ativos
    
    def _get_numeric_value(row, field_names, default=np.nan):
        st.warning("STUB: _get_numeric_value não importada.")
        return default

from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    if p_std == 0:
        return -np.inf if p_returns > risk_free_rate else (0 if p_returns == risk_free_rate else np.inf)
    return -(p_returns - risk_free_rate) / p_std

def otimizar_portfolio_scipy(retornos_df, risk_free_rate=0.02, min_weight=0.01, max_weight=0.20):
    if retornos_df.empty or len(retornos_df.columns) == 0:
        return pd.Series(dtype=float), 0, 0, 0

    num_assets = len(retornos_df.columns)
    mean_returns = retornos_df.mean()
    cov_matrix = retornos_df.cov()
    
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({


        {"type": "eq", "fun": lambda x: np.sum(x) - 1} # Soma dos pesos = 1
        },
    )             
    bounds = tuple([(min_weight, max_weight) for _ in range(num_assets)])
    initial_guess = num_assets * [1. / num_assets,]

    result = minimize(neg_sharpe_ratio, initial_guess, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = pd.Series(result.x, index=retornos_df.columns)
        # Aplicar o piso de 1% e normalizar se necessário
        optimized_weights[optimized_weights < min_weight] = 0 # Zera pesos abaixo do mínimo antes de normalizar
        # Se a soma for zero (nenhum ativo selecionado ou todos zerados), retorna vazio
        if optimized_weights.sum() == 0:
            return pd.Series(dtype=float), 0, 0, 0
            
        # Normaliza para que a soma dos pesos seja 1, respeitando o teto de 20%
        # Esta parte pode ser complexa se muitos ativos atingem o teto.
        # Uma abordagem mais simples é normalizar e depois cortar no teto, e renormalizar o restante.
        # No entanto, a otimização já deveria respeitar os bounds.
        # Vamos apenas garantir que os pesos zerados sejam realmente zero.
        optimized_weights = optimized_weights / optimized_weights.sum() # Normaliza para garantir soma 1
        
        # Re-verificar bounds após normalização (pode ser necessário se a otimização não for perfeita)
        # Por simplicidade, vamos confiar que a otimização SLSQP com bounds funciona bem.
        # Se um peso for < min_weight após otimização e normalização, ele será pequeno mas não zero.
        # A regra é que o *input* para o ativo no portfólio deve ser >= min_weight.
        # Se a otimização der um peso menor, esse ativo não entra com min_weight, ele entra com o peso otimizado (se >0)
        # ou não entra se o peso otimizado for 0.
        # A regra de min_weight é mais um bound para a otimização.

        retorno_otimizado, volatilidade_otimizada = portfolio_performance(optimized_weights, mean_returns, cov_matrix)
        sharpe_otimizado = (retorno_otimizado - risk_free_rate) / volatilidade_otimizada if volatilidade_otimizada != 0 else -np.inf
        return optimized_weights, retorno_otimizado, volatilidade_otimizada, sharpe_otimizado
    else:
        # st.warning(f"Otimização falhou: {result.message}. Retornando pesos iguais ou vazios.")
        # Fallback: se otimização falhar, pode retornar pesos iguais ou nenhum peso.
        # Para este backtest, se falhar, não alocamos.
        return pd.Series(dtype=float), 0, 0, 0


@st.cache_data(ttl=3600) # Cache por 1 hora
def obter_dados_historicos_precos_yf(ativos_e_benchmark, start_date_str, end_date_str):
    """Obtém dados de preços de fechamento ajustados do Yahoo Finance."""
    all_data = pd.DataFrame()
    for ticker in ativos_e_benchmark:
        try:
            data = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
            if not data.empty and \'Adj Close\' in data.columns:
                all_data[ticker] = data[\'Adj Close\']
            else:
                st.warning(f"Dados de fechamento ajustado não encontrados para {ticker}.")
        except Exception as e:
            st.error(f"Erro ao baixar dados para {ticker}: {e}")
    return all_data.dropna(how=\'all\')


def run_backtest(ativos_lista, benchmark_ticker, start_date, end_date, aporte_mensal, min_piotroski, max_weight_ativo, min_weight_ativo, lookback_otimizacao_meses):
    st.subheader("Executando Backtest...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    datas_rebalanceamento = pd.date_range(start_date, end_date, freq=\'MS\') # Primeiro dia de cada mês
    
    carteira_valor = aporte_mensal # Valor inicial da carteira é o primeiro aporte
    portfolio_historico_valor = pd.Series(index=datas_rebalanceamento, dtype=float)
    portfolio_holdings = {} # Para rastrear quantidade de cada ação
    cash_value = aporte_mensal
    current_asset_prices = {}
    ativos_selecionados_anualmente = {}

    # Obter todos os dados de preços de uma vez para eficiência
    todos_ativos_para_precos = list(set(ativos_lista + [benchmark_ticker]))
    status_text.text("Baixando dados históricos de preços...")
    all_prices_df = obter_dados_historicos_precos_yf(todos_ativos_para_precos, 
                                                     (datetime.strptime(start_date, \'%Y-%m-%d\') - relativedelta(months=lookback_otimizacao_meses)).strftime(\'%Y-%m-%d\'), 
                                                     end_date)
    if all_prices_df.empty:
        st.error("Não foi possível obter dados de preços. Backtest interrompido.")
        return None, None, None, None, None
    
    status_text.text("Iniciando simulação mensal...")
    total_steps = len(datas_rebalanceamento)

    for i, data_ref in enumerate(datas_rebalanceamento):
        progress_bar.progress((i + 1) / total_steps)
        status_text.text(f"Processando {data_ref.strftime(\'%Y-%m-%d\')}...")
        
        # 0. Atualizar valor da carteira com base nos preços do dia anterior ao rebalanceamento (ou do dia do aporte se for o primeiro)
        # Se não for o primeiro mês, atualiza o valor dos ativos em carteira
        if i > 0 and portfolio_holdings:
            temp_portfolio_value = 0
            data_para_preco_atual = data_ref - timedelta(days=1) # Preço do dia anterior para cálculo do valor
            while data_para_preco_atual not in all_prices_df.index and data_para_preco_atual >= all_prices_df.index.min():
                data_para_preco_atual -= timedelta(days=1)
            
            if data_para_preco_atual >= all_prices_df.index.min():
                for ticker, qty in portfolio_holdings.items():
                    if ticker in all_prices_df.columns and pd.notna(all_prices_df.loc[data_para_preco_atual, ticker]):
                        current_asset_prices[ticker] = all_prices_df.loc[data_para_preco_atual, ticker]
                        temp_portfolio_value += qty * current_asset_prices[ticker]
                    elif ticker in current_asset_prices: # Usa o último preço conhecido se o atual não estiver disponível
                        temp_portfolio_value += qty * current_asset_prices[ticker]
                carteira_valor = temp_portfolio_value + cash_value # Valor dos ativos + caixa
            # Se não houver preços, o valor da carteira não muda (exceto pelo aporte)

        # 1. Aporte Mensal (ocorre no início do mês de rebalanceamento)
        if i > 0: # Não adiciona aporte no primeiro "mês" pois já foi considerado no valor inicial
            carteira_valor += aporte_mensal
            cash_value += aporte_mensal

        portfolio_historico_valor[data_ref] = carteira_valor

        # 2. Obter dados fundamentalistas (para o ano corrente até a data de referência)
        # Idealmente, usar dados do último balanço divulgado antes de data_ref.
        # Simplificação: usar dados anuais mais recentes disponíveis.
        # A função obter_dados_fundamentalistas_detalhados_br já busca os dados mais recentes.
        dados_fund = obter_dados_fundamentalistas_detalhados_br(ativos_lista)
        if dados_fund.empty:
            status_text.text(f"Dados fundamentalistas não encontrados para {data_ref.strftime(\'%Y-%m-%d\')}. Mantendo carteira anterior se houver.")
            # Se não há dados fundamentalistas, não rebalanceia, apenas mantém o que tem e o valor é atualizado pelo aporte e preços.
            # O cash_value já foi atualizado com o aporte.
            # Atualizar o valor dos holdings com os preços mais recentes
            new_portfolio_value_assets = 0
            data_para_preco_dia = data_ref
            while data_para_preco_dia not in all_prices_df.index and data_para_preco_dia >= all_prices_df.index.min():
                 data_para_preco_dia -= timedelta(days=1)
            if data_para_preco_dia >= all_prices_df.index.min():
                for ticker, qty in portfolio_holdings.items():
                    if ticker in all_prices_df.columns and pd.notna(all_prices_df.loc[data_para_preco_dia, ticker]):
                        current_asset_prices[ticker] = all_prices_df.loc[data_para_preco_dia, ticker]
                        new_portfolio_value_assets += qty * current_asset_prices[ticker]
                    elif ticker in current_asset_prices:
                        new_portfolio_value_assets += qty * current_asset_prices[ticker]
            portfolio_historico_valor[data_ref] = new_portfolio_value_assets + cash_value
            continue

        # 3. Calcular Piotroski F-Score
        f_scores = {}
        for ticker in ativos_lista:
            if ticker in dados_fund.index:
                try:
                    # A função calcular_piotroski_f_score_br espera uma Series
                    f_score_result = calcular_piotroski_f_score_br(dados_fund.loc[ticker])
                    # A função pode retornar só o score ou (score, dict_criterios, dict_valores)
                    if isinstance(f_score_result, tuple):
                        f_scores[ticker] = f_score_result[0]
                    else:
                        f_scores[ticker] = f_score_result
                except Exception as e:
                    # st.warning(f"Erro ao calcular Piotroski para {ticker} em {data_ref.strftime(\'%Y-%m-%d\')}: {e}")
                    f_scores[ticker] = 0
            else:
                f_scores[ticker] = 0
        
        # 4. Filtrar ativos com F-Score >= min_piotroski
        ativos_elegiveis = [ticker for ticker, score in f_scores.items() if score >= min_piotroski]

        if not ativos_elegiveis:
            status_text.text(f"Nenhum ativo elegível em {data_ref.strftime(\'%Y-%m-%d\')}. Mantendo caixa.")
            # Se não há ativos elegíveis, todo o valor da carteira (que já inclui o novo aporte) vira/continua caixa.
            # Os ativos que estavam em carteira são "vendidos" (simbolicamente, o valor deles é transferido para o caixa)
            # e o portfolio_holdings é zerado.
            # O valor da carteira já foi atualizado. O cash_value se torna o carteira_valor e holdings zeram.
            cash_value = portfolio_historico_valor[data_ref] # Todo valor vira caixa
            portfolio_holdings = {} # Zera posições
            continue

        # 5. Obter retornos históricos para otimização
        # Usar dados de preços até um dia antes da data de rebalanceamento
        end_date_otimizacao = data_ref - timedelta(days=1)
        start_date_otimizacao = end_date_otimizacao - relativedelta(months=lookback_otimizacao_meses)
        
        # Filtrar all_prices_df para o período e ativos corretos
        precos_otimizacao = all_prices_df.loc[start_date_otimizacao:end_date_otimizacao, ativos_elegiveis].copy()
        precos_otimizacao.dropna(axis=1, how=\'any\', inplace=True) # Remove ativos sem dados suficientes no período
        
        if precos_otimizacao.shape[0] < 2 or precos_otimizacao.shape[1] == 0: # Precisa de pelo menos 2 pontos para pct_change e pelo menos 1 ativo
            status_text.text(f"Dados de retorno insuficientes para otimização em {data_ref.strftime(\'%Y-%m-%d\')}. Mantendo caixa/carteira anterior.")
            cash_value = portfolio_historico_valor[data_ref]
            portfolio_holdings = {}
            continue
            
        retornos_otimizacao = precos_otimizacao.pct_change().dropna(how=\'all\')
        
        if retornos_otimizacao.empty or len(retornos_otimizacao.columns) == 0:
            status_text.text(f"Retornos para otimização não puderam ser calculados em {data_ref.strftime(\'%Y-%m-%d\')}. Mantendo caixa/carteira anterior.")
            cash_value = portfolio_historico_valor[data_ref]
            portfolio_holdings = {}
            continue

        # 6. Otimizar portfólio
        # Usar taxa livre de risco Selic (aproximada) ou um valor fixo
        # Para simplificar, usaremos a taxa default da função, mas idealmente seria dinâmica
        pesos_otimizados, _, _, _ = otimizar_portfolio_scipy(retornos_otimizacao, 
                                                                  risk_free_rate=0.02, # Exemplo, idealmente buscar Selic do período
                                                                  min_weight=min_weight_ativo, 
                                                                  max_weight=max_weight_ativo)

        if pesos_otimizados.empty or pesos_otimizados.sum() == 0:
            status_text.text(f"Otimização não retornou pesos válidos em {data_ref.strftime(\'%Y-%m-%d\')}. Mantendo caixa/carteira anterior.")
            cash_value = portfolio_historico_valor[data_ref]
            portfolio_holdings = {}
            continue

        # 7. Simular alocação e registrar
        # 

# Regra: Não há venda, apenas compra e rebalanceamento dos pesos.
        # O "rebalanceamento" aqui significa que, a cada mês, a carteira é liquidada (valor vira caixa)
        # e uma nova carteira é comprada com base nos novos pesos e no valor total atualizado (incluindo novo aporte).
        
        # Zerar holdings atuais, pois vamos "recomprar" tudo com os novos pesos
        # O valor total da carteira (incluindo o aporte do mês) será usado para comprar os novos ativos.
        # O cash_value no início desta etapa de alocação é o valor total disponível para investir.
        cash_to_invest_this_month = portfolio_historico_valor[data_ref] # Este já inclui o aporte
        new_portfolio_holdings = {}
        spent_on_assets = 0

        # Obter preços do dia do rebalanceamento para simular a compra
        data_para_compra = data_ref
        while data_para_compra not in all_prices_df.index and data_para_compra < all_prices_df.index.max():
            data_para_compra += timedelta(days=1) # Tenta o próximo dia útil se o dia do rebal. não tiver preço
        
        if data_para_compra > all_prices_df.index.max(): # Se não achou preço para frente, tenta para trás (menos ideal)
            data_para_compra = data_ref
            while data_para_compra not in all_prices_df.index and data_para_compra >= all_prices_df.index.min():
                data_para_compra -= timedelta(days=1)
        
        # Garantir que temos preços para os ativos selecionados
        precos_compra_validos = True
        if data_para_compra < all_prices_df.index.min() or data_para_compra > all_prices_df.index.max():
            precos_compra_validos = False
        else:
            for ticker in pesos_otimizados.index:
                if pesos_otimizados[ticker] > 0 and (ticker not in all_prices_df.columns or pd.isna(all_prices_df.loc[data_para_compra, ticker])):
                    # st.warning(f"Preço não disponível para {ticker} em {data_para_compra.strftime(\"%Y-%m-%d\")} para compra. Ativo será ignorado neste mês.")
                    pesos_otimizados[ticker] = 0 # Ignora ativo se não tem preço
            if pesos_otimizados.sum() > 0:
                pesos_otimizados = pesos_otimizados / pesos_otimizados.sum() # Renormaliza se algum ativo foi removido
            else:
                precos_compra_validos = False

        if not precos_compra_validos or pesos_otimizados.empty or pesos_otimizados.sum() == 0:
            status_text.text(f"Não foi possível obter preços para alocação em {data_ref.strftime(\"%Y-%m-%d\")}. Mantendo caixa.")
            cash_value = cash_to_invest_this_month
            portfolio_holdings = {}
        else:
            current_year = data_ref.year
            if current_year not in ativos_selecionados_anualmente:
                ativos_selecionados_anualmente[current_year] = []
            
            ativos_pesos_mes = {}
            for ticker, weight in pesos_otimizados.items():
                if weight > 0:
                    valor_a_investir_ativo = cash_to_invest_this_month * weight
                    preco_ativo = all_prices_df.loc[data_para_compra, ticker]
                    if pd.notna(preco_ativo) and preco_ativo > 0:
                        quantidade_comprada = valor_a_investir_ativo / preco_ativo
                        new_portfolio_holdings[ticker] = new_portfolio_holdings.get(ticker, 0) + quantidade_comprada
                        spent_on_assets += valor_a_investir_ativo
                        ativos_pesos_mes[ticker] = weight
                        current_asset_prices[ticker] = preco_ativo # Atualiza o último preço conhecido
                    else:
                        # st.warning(f"Preço inválido para {ticker} em {data_para_compra.strftime(\"%Y-%m-%d\")}. Não foi possível comprar.")
                        pass # O valor destinado a este ativo permanece em caixa
            
            ativos_selecionados_anualmente[current_year].append({
                "Data": data_ref.strftime("%Y-%m-%d"),
                "Ativos_Pesos": ativos_pesos_mes
            })
            portfolio_holdings = new_portfolio_holdings
            cash_value = cash_to_invest_this_month - spent_on_assets

    progress_bar.empty()
    status_text.text("Backtest concluído!")

    # Calcular valor final da carteira
    valor_final_portfolio = 0
    data_final_para_preco = end_date
    if isinstance(data_final_para_preco, str):
        data_final_para_preco = datetime.strptime(end_date, 
imazole("%Y-%m-%d"))
    
    # Achar o último dia com preços disponíveis até a data final do backtest
    while data_final_para_preco not in all_prices_df.index and data_final_para_preco >= all_prices_df.index.min():
        data_final_para_preco -= timedelta(days=1)

    if data_final_para_preco >= all_prices_df.index.min():
        for ticker, qty in portfolio_holdings.items():
            if ticker in all_prices_df.columns and pd.notna(all_prices_df.loc[data_final_para_preco, ticker]):
                valor_final_portfolio += qty * all_prices_df.loc[data_final_para_preco, ticker]
            elif ticker in current_asset_prices: # Fallback para último preço conhecido
                 valor_final_portfolio += qty * current_asset_prices[ticker]
        valor_final_portfolio += cash_value
        # Atualizar o último valor no histórico do portfólio
        if not portfolio_historico_valor.empty:
            last_rebal_date = portfolio_historico_valor.index[-1]
            # Se a data final do backtest for diferente da última data de rebalanceamento,
            # precisamos adicionar um ponto para o valor final na data exata de término.
            # No entanto, portfolio_historico_valor já está indexado pelas datas de rebalanceamento.
            # O valor em portfolio_historico_valor[last_rebal_date] já reflete o valor *após* o aporte e *antes* da alocação daquele mês.
            # O cálculo correto do valor final já foi feito acima.
            # Para o gráfico, é melhor ter o valor na data de término.
            # Vamos adicionar um ponto final se a data_final_para_preco for posterior à última data de rebalanceamento.
            if data_final_para_preco > last_rebal_date:
                 portfolio_historico_valor[data_final_para_preco] = valor_final_portfolio
            else: # Atualiza o valor da última data de rebalanceamento com o valor de fechamento
                 portfolio_historico_valor[last_rebal_date] = valor_final_portfolio
    else: # Se não há preços na data final, usa o último valor calculado na última data de rebalanceamento
        if not portfolio_historico_valor.empty:
            valor_final_portfolio = portfolio_historico_valor.iloc[-1]

    # Benchmark
    status_text.text("Calculando benchmark...")
    benchmark_prices = all_prices_df[[benchmark_ticker]].loc[datetime.strptime(start_date, 
imazole("%Y-%m-%d")):datetime.strptime(end_date, 
imazole("%Y-%m-%d"))].copy()
    benchmark_prices.dropna(inplace=True)
    
    benchmark_historico_valor = pd.Series(index=benchmark_prices.index, dtype=float)
    if not benchmark_prices.empty:
        initial_benchmark_price = benchmark_prices[benchmark_ticker].iloc[0]
        shares_benchmark = aporte_mensal / initial_benchmark_price # Aporte inicial
        current_benchmark_value = aporte_mensal
        benchmark_historico_valor.iloc[0] = current_benchmark_value
        
        # Simular aportes mensais no benchmark
        # A cada data de rebalanceamento da nossa carteira, fazemos um aporte no benchmark
        for data_ref in datas_rebalanceamento: # Essas são as datas de aporte
            if data_ref == datas_rebalanceamento[0]: # Primeiro aporte já feito
                benchmark_historico_valor[data_ref] = current_benchmark_value
                continue

            # Encontrar o preço do benchmark mais próximo da data de aporte
            data_compra_benchmark = data_ref
            while data_compra_benchmark not in benchmark_prices.index and data_compra_benchmark < benchmark_prices.index.max():
                data_compra_benchmark += timedelta(days=1)
            while data_compra_benchmark not in benchmark_prices.index and data_compra_benchmark > benchmark_prices.index.min():
                data_compra_benchmark -= timedelta(days=1)
            
            if data_compra_benchmark in benchmark_prices.index:
                current_price_benchmark = benchmark_prices.loc[data_compra_benchmark, benchmark_ticker]
                # Atualiza valor das cotas existentes
                current_benchmark_value = shares_benchmark * current_price_benchmark
                # Novo aporte
                current_benchmark_value += aporte_mensal
                shares_benchmark = current_benchmark_value / current_price_benchmark # Novas cotas totais
                benchmark_historico_valor[data_ref] = current_benchmark_value
            else: # Se não achar preço, repete o valor anterior (ou o valor com aporte se for o caso)
                # Isso é uma simplificação, idealmente o aporte seria feito no próximo dia com preço
                if data_ref in benchmark_historico_valor.index:
                     benchmark_historico_valor[data_ref] = benchmark_historico_valor.get(data_ref - pd.Timedelta(days=1), current_benchmark_value) + aporte_mensal
                else: # Se for uma data nova, preenche com o valor anterior + aporte
                    prev_date = benchmark_historico_valor.index[benchmark_historico_valor.index < data_ref].max()
                    if pd.notna(prev_date):
                        benchmark_historico_valor[data_ref] = benchmark_historico_valor[prev_date] + aporte_mensal
                    else:
                        benchmark_historico_valor[data_ref] = current_benchmark_value + aporte_mensal # Should not happen after first aporte
                # Atualiza current_benchmark_value para o próximo loop
                current_benchmark_value = benchmark_historico_valor[data_ref]

        # Preencher NaNs no benchmark_historico_valor com o último valor válido (ffill)
        benchmark_historico_valor = benchmark_historico_valor.ffill()
        # Garantir que o benchmark_historico_valor cubra todo o período do portfolio_historico_valor para comparação
        # Reindexar e preencher para frente
        if not portfolio_historico_valor.empty:
            benchmark_historico_valor = benchmark_historico_valor.reindex(portfolio_historico_valor.index, method=\"ffill\").fillna(method=\"bfill\")

    status_text.empty()
    return portfolio_historico_valor, benchmark_historico_valor, calcular_metricas(portfolio_historico_valor, aporte_mensal * len(datas_rebalanceamento)), calcular_metricas(benchmark_historico_valor, aporte_mensal * len(datas_rebalanceamento)) if not benchmark_historico_valor.empty else None, ativos_selecionados_anualmente

def calcular_metricas(series_valor, total_investido):
    if series_valor.empty or series_valor.isnull().all() or len(series_valor) < 2:
        return {"Retorno Total": 0, "Retorno Anualizado": 0, "Volatilidade Anualizada": 0, "Sharpe Ratio": 0, "Drawdown Máximo": 0, "Valor Final": 0}

    retorno_total_percentual = (series_valor.iloc[-1] / series_valor.iloc[0]) - 1 
    # Correção: Retorno total sobre o capital investido, não sobre o valor inicial que é só o primeiro aporte.
    # O cálculo do retorno total deve considerar o valor final sobre o total de aportes.
    # No entanto, para retorno anualizado, usamos a série de valor.
    # Para Sharpe, usamos retornos diários/mensais da série de valor.

    # Retorno Anualizado (Geométrico)
    num_anos = (series_valor.index[-1] - series_valor.index[0]).days / 365.25
    retorno_anualizado = (series_valor.iloc[-1] / series_valor.iloc[0])**(1/num_anos) - 1 if num_anos > 0 else 0

    # Volatilidade Anualizada
    retornos_diarios = series_valor.pct_change().dropna()
    if not retornos_diarios.empty:
        # Assumindo que a série de valor é mensal (rebalanceamento mensal)
        # Se for diária, sqrt(252). Se for mensal, sqrt(12).
        # Como portfolio_historico_valor é mensal (indexado por datas_rebalanceamento)
        volatilidade_anualizada = retornos_diarios.std() * np.sqrt(12) 
    else:
        volatilidade_anualizada = 0

    # Sharpe Ratio Anualizado (usando taxa risk-free de 0% para simplificar, ou Selic média do período)
    risk_free_rate_anual = 0.0 # Simplificado
    # Média dos retornos mensais * 12 para anualizar o retorno bruto
    retorno_medio_anualizado_serie = retornos_diarios.mean() * 12 if not retornos_diarios.empty else 0
    sharpe_ratio = (retorno_medio_anualizado_serie - risk_free_rate_anual) / volatilidade_anualizada if volatilidade_anualizada != 0 else 0

    # Drawdown Máximo
    rolling_max = series_valor.cummax()
    drawdown = series_valor/rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "Retorno Total sobre Investimento": (series_valor.iloc[-1] - total_investido) / total_investido if total_investido > 0 else 0,
        "Retorno Anualizado (CAGR)": retorno_anualizado,
        "Volatilidade Anualizada": volatilidade_anualizada,
        "Sharpe Ratio (Anualizado)": sharpe_ratio,
        "Drawdown Máximo": max_drawdown,
        "Valor Final": series_valor.iloc[-1],
        "Total Investido": total_investido
    }

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Backtest de Carteira de Ações B3 com Piotroski F-Score e Otimização de Sharpe")

# Sidebar para inputs
st.sidebar.header("Parâmetros do Backtest")

start_date_default = "2018-01-01"
end_date_default = datetime.today().strftime("%Y-%m-%d")

start_date_str = st.sidebar.text_input("Data de Início (YYYY-MM-DD)", start_date_default)
end_date_str = st.sidebar.text_input("Data de Fim (YYYY-MM-DD)", end_date_default)

aporte_mensal_input = st.sidebar.number_input("Aporte Mensal (R$)", min_value=100, value=1000, step=100)

ativos_padrao = "AGRO3.SA, B3SA3.SA, BBAS3.SA, BBSE3.SA, BPAC11.SA, EGIE3.SA, ITUB3.SA, PRIO3.SA, PSSA3.SA, SAPR4.SA, SBSP3.SA, VIVT3.SA, WEGE3.SA, TOTS3.SA, TAEE3.SA, CMIG3.SA"
st.sidebar.markdown("**Ativos para Backtest (separados por vírgula):**")
ativos_input_str = st.sidebar.text_area("Lista de Ativos", ativos_padrao, height=150)
ativos_lista_input = [ticker.strip().upper() for ticker in ativos_input_str.split(",") if ticker.strip()]

benchmark_ticker_input = st.sidebar.text_input("Ticker do Benchmark", "BOVA11.SA").strip().upper()

st.sidebar.markdown("--- Configuracoes Avancadas ---")
min_piotroski_input = st.sidebar.slider("Piotroski F-Score Mínimo", min_value=0, max_value=9, value=7, step=1)
max_weight_ativo_input = st.sidebar.slider("Alocação Máxima por Ativo (%)", min_value=5, max_value=100, value=20, step=1) / 100.0
min_weight_ativo_input = st.sidebar.slider("Alocação Mínima por Ativo (se selecionado) (%)", min_value=1, max_value=10, value=1, step=1) / 100.0
lookback_otimizacao_input = st.sidebar.slider("Janela para Otimização (meses)", min_value=6, max_value=36, value=12, step=1)

if st.sidebar.button("Iniciar Backtest"):
    if not ativos_lista_input:
        st.error("Por favor, insira ao menos um ativo para o backtest.")
    elif not benchmark_ticker_input:
        st.error("Por favor, insira o ticker do benchmark.")
    else:
        try:
            datetime.strptime(start_date_str, 
imazole("%Y-%m-%d"))
            datetime.strptime(end_date_str, 
imazole("%Y-%m-%d"))
        except ValueError:
            st.error("Formato de data inválido. Use YYYY-MM-DD.")
            st.stop()

        portfolio_values, benchmark_values, portfolio_metrics, benchmark_metrics, ativos_por_ano = run_backtest(
            ativos_lista_input,
            benchmark_ticker_input,
            start_date_str,
            end_date_str,
            aporte_mensal_input,
            min_piotroski_input,
            max_weight_ativo_input,
            min_weight_ativo_input,
            lookback_otimizacao_input
        )

        if portfolio_values is not None and not portfolio_values.empty:
            st.subheader("Evolução do Valor da Carteira vs. Benchmark")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values, mode=\"lines\", name=\"Carteira Otimizada\"))
            if benchmark_values is not None and not benchmark_values.empty:
                fig.add_trace(go.Scatter(x=benchmark_values.index, y=benchmark_values, mode=\"lines\", name=f"Benchmark ({benchmark_ticker_input})\"))
            fig.update_layout(title=\"Valor da Carteira ao Longo do Tempo\", xaxis_title=\"Data\", yaxis_title=\"Valor (R$)\", legend_title=\"Legenda\")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Métricas Finais do Portfólio")
            if portfolio_metrics:
                # Formatar métricas para exibição
                metrics_df = pd.DataFrame([portfolio_metrics])
                metrics_df = metrics_df.rename(columns={
                    "Retorno Total sobre Investimento": "Retorno Total s/ Investimento (%)",
                    "Retorno Anualizado (CAGR)": "Retorno Anualizado (CAGR) (%)",
                    "Volatilidade Anualizada": "Volatilidade Anualizada (%)",
                    "Drawdown Máximo": "Drawdown Máximo (%)"
                })
                # Converter para percentual
                for col_pct in ["Retorno Total s/ Investimento (%)", "Retorno Anualizado (CAGR) (%)", "Volatilidade Anualizada (%)", "Drawdown Máximo (%)"]:
                    if col_pct in metrics_df.columns:
                         metrics_df[col_pct] = (metrics_df[col_pct] * 100).round(2).astype(str) + \"%\"
                metrics_df["Sharpe Ratio (Anualizado)"] = metrics_df["Sharpe Ratio (Anualizado)"].round(2)
                metrics_df["Valor Final"] = metrics_df["Valor Final"].round(2).apply(lambda x: f"R$ {x:,.2f}".replace(\",\", \"X\").replace(\".\", \",\").replace(\"X\", \".\"))
                metrics_df["Total Investido"] = metrics_df["Total Investido"].round(2).apply(lambda x: f"R$ {x:,.2f}".replace(\",\", \"X\").replace(\".\", \",\").replace(\"X\", \".\"))
                st.table(metrics_df.T.rename(columns={0: "Carteira"}))
            
            if benchmark_metrics:
                st.subheader(f"Métricas Finais do Benchmark ({benchmark_ticker_input})")
                metrics_bench_df = pd.DataFrame([benchmark_metrics])
                metrics_bench_df = metrics_bench_df.rename(columns={
                    "Retorno Total sobre Investimento": "Retorno Total s/ Investimento (%)",
                    "Retorno Anualizado (CAGR)": "Retorno Anualizado (CAGR) (%)",
                    "Volatilidade Anualizada": "Volatilidade Anualizada (%)",
                    "Drawdown Máximo": "Drawdown Máximo (%)"
                })
                for col_pct in ["Retorno Total s/ Investimento (%)", "Retorno Anualizado (CAGR) (%)", "Volatilidade Anualizada (%)", "Drawdown Máximo (%)"]:
                    if col_pct in metrics_bench_df.columns:
                        metrics_bench_df[col_pct] = (metrics_bench_df[col_pct] * 100).round(2).astype(str) + \"%\"
                metrics_bench_df["Sharpe Ratio (Anualizado)"] = metrics_bench_df["Sharpe Ratio (Anualizado)"].round(2)
                metrics_bench_df["Valor Final"] = metrics_bench_df["Valor Final"].round(2).apply(lambda x: f"R$ {x:,.2f}".replace(\",\", \"X\").replace(\".\", \",\").replace(\"X\", \".\"))
                metrics_bench_df["Total Investido"] = metrics_bench_df["Total Investido"].round(2).apply(lambda x: f"R$ {x:,.2f}".replace(\",\", \"X\").replace(\".\", \",\").replace(\"X\", \".\"))
                st.table(metrics_bench_df.T.rename(columns={0: "Benchmark"}))

            st.subheader("Ativos Selecionados e Pesos por Ano/Mês")
            if ativos_por_ano:
                for ano, meses_data in ativos_por_ano.items():
                    st.markdown(f"**Ano: {ano}**")
                    all_monthly_allocations = []
                    for entrada_mes in meses_data:
                        data_mes = entrada_mes["Data"]
                        pesos_mes = entrada_mes["Ativos_Pesos"]
                        if pesos_mes: # Só mostra se houve alocação
                            df_pesos_mes = pd.DataFrame(list(pesos_mes.items()), columns=["Ativo", "Peso (%)"])
                            df_pesos_mes["Peso (%)"] = (df_pesos_mes["Peso (%)"] * 100).round(2)
                            df_pesos_mes["Data"] = data_mes
                            all_monthly_allocations.append(df_pesos_mes)
                    
                    if all_monthly_allocations:
                        df_ano_completo = pd.concat(all_monthly_allocations).set_index(["Data", "Ativo"])
                        st.dataframe(df_ano_completo)
                    else:
                        st.markdown("_Nenhuma alocação neste ano._")
            else:
                st.info("Nenhum ativo foi selecionado durante o período do backtest com os critérios definidos.")
        else:
            st.error("Backtest não pôde ser concluído. Verifique os logs ou parâmetros.")

st.sidebar.markdown("--- Sobre ---")
st.sidebar.info("Este aplicativo realiza um backtest de uma estratégia de investimento baseada no Piotroski F-Score e otimização de portfólio para maximizar o Índice de Sharpe. Desenvolvido como parte de uma simulação.")

# Para rodar este app: streamlit run backtest_streamlit_app.py

