# Cálculo de distâncias e vetores
import numpy as np

# Traçagem de gráficos
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Carregar e lidar com o dataset
import pandas as pd
from sklearn.datasets import load_iris

# Como este dataset possui pouquíssimos outliers e não é o foco,
# já podemos lidar com a parte algorítmica para a classificação
class K_Means:
    # Neste problema, sabemos que existem 3 espécies, portanto k=3
    # Para tolerância e limite de iterações, utilizaremos valores comumente usados
    def __init__(self, k=3, tolerancia=0.001, maxIteracoes=300):
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
                distancias = [np.linalg.norm(
                    featureSet - self.centroides[centroide])
                    for centroide in self.centroides]

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
    iris = load_iris()
    dfIris = pd.DataFrame(iris['data'], columns=iris['feature_names'])

    # Separa dados em treino e teste
    mask = np.random.rand(len(dfIris)) < 0.8
    dadosTreino = dfIris[mask].values
    dadosTeste = dfIris[~mask].values

    # Cores dos clusters
    cores = ["red", "green", "blue"]

    # Temos 4 features: comprimento e largura das sépalas e pétalas
    # Vamos aglomerar dados de sépalas e pétalas separadamente
    # Visualizando os dados
    plt.scatter(dfIris.values[:,0], dfIris.values[:,1], marker='.')
    plt.xlabel("Comprimento sépala")
    plt.ylabel("Largura sépala")
    plt.title("IRIS")
    plt.show()

    plt.scatter(dfIris.values[:,2], dfIris.values[:,3], marker='.')
    plt.xlabel("Comprimento pétala")
    plt.ylabel("Largura pétala")
    plt.title("IRIS")
    plt.show()

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
