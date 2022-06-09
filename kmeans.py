# Cálculo de distâncias e vetores
import numpy as np

# Traçagem de gráficos
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Carregar e lidar com o dataset
import pandas as pd
from sklearn.datasets import load_digits

class K_Means:
    # Neste problema, sabemos que existem 10 dígitos, portanto k=10
    # Para tolerância e limite de iterações, utilizaremos valores comumente usados
    def __init__(self, k=10, tolerancia=0.001, maxIteracoes=300):
        self.k = k
        self.tolerancia = tolerancia
        self.maxIteracoes = maxIteracoes

    # Ajusta o modelo aos dados fornecidos
    def fit(self, data):
        # Inicializa o dicionário de centroides
        self.centroides = {}

        # Inicializa posição de centroides aleatoriamente
        np.random.shuffle(data)
        for i in range(self.k):
            self.centroides[i] = data[i]

        for i in range(self.maxIteracoes):
            # Limpa classificação dos centroides a cada iteração (movimentação)
            self.classificacoes = {}

            # Inicializa dicionário de classificações (cada chave é um centroide)
            for i in range(self.k):
                self.classificacoes[i] = []

            for featureSet in data:
                # Popula a lista com distâncias entre cada observações e os k centroides
                distancias = [np.linalg.norm(featureSet - self.centroides[centroide]) for centroide in self.centroides]

                # Determina a classificação através da menor distância
                classificacao = distancias.index(min(distancias))
                # Adiciona a observação à chave correspondente ao centroide no dicionário de classificações
                self.classificacoes[classificacao].append(featureSet)

            # Será utilizado para comparação futuramente
            centroideAnterior = dict(self.centroides)

            # Tira a média das classificações para achar a nova posição de cada centroide
            for classificacao in self.classificacoes:
                self.centroides[classificacao] = np.average(
                    self.classificacoes[classificacao], axis=0)

            # Verifica se os centroides estão se movendo a cada nova iteração
            # Se não estiverem, encerra o algoritmo
            otimizado = True
            for c in self.centroides:
                centroideOriginal = centroideAnterior[c]
                centroideAtual = self.centroides[c]
                diferencaPosicao = np.sum((abs(centroideAtual-centroideOriginal)) / centroideOriginal * 100.0)
                # print(diferencaPosicao)
                if diferencaPosicao > self.tolerancia:
                    otimizado = False

            if otimizado:
                break

    # Categoriza novos dados de acordo com o modelo já ajustado
    def predict(self, data):
        distancias = [np.linalg.norm(data-self.centroides[centroide])
                    for centroide in self.centroides]
        classificacao = distancias.index(min(distancias))
        return classificacao


if __name__ == '__main__':
    # Primeiramente, carregamos o dataset da íris
    digits = load_digits()
    dfDigits = pd.DataFrame(digits['data'], columns=digits['feature_names'])

    # Separa dados em treino e teste
    mask = np.random.rand(len(dfDigits)) < 0.8
    dadosTreino = dfDigits[mask].values
    dadosTeste = dfDigits[~mask].values

    # Cores dos clusters
    cores = ["red", "green", "blue"]

    # Temos 64 features: cada pixel na imagem
    # Visualizando os dados
    plt.scatter(dfDigits.values[:,0], dfDigits.values[:,1], marker='.')
    plt.xlabel("Comprimento sépala")
    plt.ylabel("Largura sépala")
    plt.title("DIGITS")
    plt.show()

    # plt.scatter(dfDigits.values[:,2], dfDigits.values[:,3], marker='.')
    # plt.xlabel("Comprimento pétala")
    # plt.ylabel("Largura pétala")
    # plt.title("DIGITS")
    # plt.show()

    # Instancia KMeans e ajusta modelo aos dados de treino
    clf = K_Means()
    clf.fit(dadosTreino)

    ### SÉPALAS
    # Traça centroides
    for centroide in clf.centroides:
        plt.scatter(clf.centroides[centroide][0], clf.centroides[centroide]
                    [1], marker='o', color='black', s=150, linewidths=5)

    # Traça pontos de treino
    for classificacao in clf.classificacoes:
        cor = cores[classificacao]
        for featureSet in clf.classificacoes[classificacao]:
            plt.scatter(featureSet[0], featureSet[1],
                    marker='x', color=cor, s=50, linewidths=5)

    # Traça pontos de teste
    for desconhecido in dadosTeste:
        classificacao = clf.predict(desconhecido)
        plt.scatter(desconhecido[0], desconhecido[1], marker="*", color=cores[classificacao])

    # Legenda
    legendaTreino = Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=7, label='Treino')
    legendaTeste = Line2D([], [], color='black', marker='*', linestyle='None',
                          markersize=6, label='Teste')
    plt.legend(handles=[legendaTreino, legendaTeste])

    plt.xlabel("Comprimento sépala")
    plt.ylabel("Largura sépala")
    plt.title("Dataset Íris clusterizado")
    plt.show()

    ### PÉTALAS
    # Traça centroides
    for centroide in clf.centroides:
        plt.scatter(clf.centroides[centroide][2], clf.centroides[centroide]
                    [3], marker='o', color='black', s=150, linewidths=5)

    # Traça pontos de treino
    for classificacao in clf.classificacoes:
        cor = cores[classificacao]
        for featureSet in clf.classificacoes[classificacao]:
            plt.scatter(featureSet[2], featureSet[3],
                    marker='x', color=cor, s=50, linewidths=5)

    # Traça pontos de teste
    for desconhecido in dadosTeste:
        classificacao = clf.predict(desconhecido)
        plt.scatter(desconhecido[2], desconhecido[3], marker="*", color=cores[classificacao])

    # Legenda
    legendaTreino = Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=7, label='Treino')
    legendaTeste = Line2D([], [], color='black', marker='*', linestyle='None',
                          markersize=6, label='Teste')
    plt.legend(handles=[legendaTreino, legendaTeste])

    plt.xlabel("Comprimento pétala")
    plt.ylabel("Largura pétala")
    plt.title("Dataset Íris clusterizado")

    plt.show()
