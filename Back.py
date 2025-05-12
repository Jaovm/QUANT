import pandas as pd
import yfinance as yf
from datetime import datetime
from financial_analyzer_enhanced_corrected import (
    calcular_piotroski_f_score_br,
    obter_dados_fundamentalistas_detalhados_br,
    otimizar_portfolio_scipy
)
import matplotlib.pyplot as plt
import numpy as np

# Parâmetros
start_date = "2018-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
tickers = [  # Lista reduzida para performance (use a lista completa se preferir)
    "ABEV3.SA", "ITUB4.SA", "VALE3.SA", "PETR4.SA", "WEGE3.SA", "B3SA3.SA",
    "BBAS3.SA", "PRIO3.SA", "RADL3.SA", "HAPV3.SA", "RENT3.SA", "LREN3.SA"
]
benchmark = "BOVA11.SA"

# Download de preços
data = yf.download(tickers + [benchmark], start=start_date, end=end_date)["Adj Close"]
data = data.dropna(how='all')
data = data.fillna(method='ffill')

# Inicialização
resultados = []
carteira_valor = pd.Series(dtype=float)
benchmark_valor = pd.Series(dtype=float)
capital_inicial = 1_000.0
capital = capital_inicial
capital_bench = capital_inicial
rebalance_dates = data.resample("M").last().index

# Loop de rebalanceamento mensal
for rebalance_date in rebalance_dates:
    subset = data[:rebalance_date].dropna(axis=1, thresh=50)
    if subset.empty or len(subset.columns) < 3:
        continue

    tickers_validos = list(subset.columns)
    
    # Obter fundamentos
    fundamentos = obter_dados_fundamentalistas_detalhados_br(tickers_validos)
    fundamentos = preencher_campos_faltantes_brapi(fundamentos)

    # Calcular F-Score
    fundamentos = calcular_piotroski_f_score_br(fundamentos)
    filtrados = fundamentos[fundamentos['f_score'] >= 7]
    tickers_finais = list(filtrados['ticker'])

    if len(tickers_finais) < 3:
        continue

    # Preços para otimização
    returns = subset[tickers_finais].pct_change().dropna()
    if returns.isnull().values.any():
        continue

    # Otimização
    pesos = otimizar_portfolio_scipy(
        returns,
        bounds=(0.01, 0.2),
        risk_free_rate=0.13 / 12  # CDI médio mensal
    )
    pesos = dict(zip(tickers_finais, pesos))

    # Alocação
    ultima_data = rebalance_date
    if ultima_data not in data.index:
        continue
    precos_finais = data.loc[ultima_data, list(pesos.keys())]
    alocacao_valores = {ticker: capital * peso for ticker, peso in pesos.items()}
    quantidades = {ticker: alocacao_valores[ticker] / precos_finais[ticker] for ticker in pesos}

    # Projeção até o próximo rebalanceamento
    proxima_data = rebalance_date + pd.DateOffset(months=1)
    futuro = data.loc[rebalance_date:proxima_data, list(pesos.keys())]
    for dt in futuro.index:
        valor_total = sum(futuro.loc[dt, t] * quantidades[t] for t in pesos)
        carteira_valor.loc[dt] = valor_total

    # Benchmark
    bova = data[benchmark].loc[rebalance_date:proxima_data]
    for dt in bova.index:
        benchmark_valor.loc[dt] = capital_bench * (bova.loc[dt] / bova.iloc[0])

    capital = carteira_valor.iloc[-1]
    capital_bench = benchmark_valor.iloc[-1]
    resultados.append({"data": rebalance_date.date(), "tickers": pesos})

# Métricas
def calcular_metricas(series):
    retornos = series.pct_change().dropna()
    retorno_anual = (series.iloc[-1] / series.iloc[0]) ** (1 / (len(series) / 252)) - 1
    vol_anual = retornos.std() * np.sqrt(252)
    sharpe = retorno_anual / vol_anual
    drawdown = (series / series.cummax() - 1).min()
    return retorno_anual, vol_anual, sharpe, drawdown

ret, vol, sharpe, dd = calcular_metricas(carteira_valor)
ret_b, vol_b, sharpe_b, dd_b = calcular_metricas(benchmark_valor)

# Gráfico
plt.figure(figsize=(12,6))
plt.plot(carteira_valor, label='Carteira Otimizada')
plt.plot(benchmark_valor, label='BOVA11')
plt.title("Evolução do Valor da Carteira")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("evolucao_carteira.png", dpi=300)

# Resultados
print(f"Retorno anualizado: {ret:.2%}")
print(f"Volatilidade anualizada: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Drawdown Máximo: {dd:.2%}")
print("\nComparação com BOVA11:")
print(f"Retorno anualizado: {ret_b:.2%}")
print(f"Volatilidade anualizada: {vol_b:.2%}")
print(f"Sharpe Ratio: {sharpe_b:.2f}")
print(f"Drawdown Máximo: {dd_b:.2%}")

# Lista anual de ativos
pd.DataFrame(resultados).to_csv("ativos_por_mes.csv", index=False)
