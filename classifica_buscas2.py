
from collections import Counter

import pandas as pd
data_frame = pd.read_csv('buscas2.csv')

X_df=data_frame[['home', 'busca', 'logado']]
Y_df=data_frame['comprou']

#pega os dummies da coluna alfanumerica 'busca'
Xdummies_df = pd.get_dummies(X_df)

#Quando o array jah eh binario ao rodar o metodo get_dummies ele retorna duas colunas onde a segunda coluna eh o valor original, logo outra abordagem eh: Ydummies = pd.get_dummies(Y)[1], onde 1 eh o titulo da coluna
Ydummies_df = Y_df


X = Xdummies_df.values
Y = Ydummies_df.values


acerto_de_um = len(Y[Y==1])
acerto_de_zero = len(Y[Y==0])




porcentagem_treino = 0.8
porcentagem_teste = 0.1


tamanho_de_treino = int( porcentagem_treino * len(Y) )
tamanho_de_teste = int( porcentagem_teste * len(Y) )
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

# 0 ate 799
treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

# 800 ate 899
fim_de_teste = (tamanho_de_treino+tamanho_de_teste)
teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]
#fim_de_teste = fim_de_treino + tamanho_de_treino

#900 ate 999
validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:] 


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes,teste_dados, teste_marcacoes):

    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    #diferencas = resultado - teste_marcacoes

    #acertos = [d for d in diferencas if d == 0]
    acertos = (resultado==teste_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do {0}: {1}".format(nome,taxa_de_acerto)
    print(msg)
    return taxa_de_acerto


from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaboostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomial>resultadoAdaBoost:
   vencedor = modeloMultinomial
else:
   vencedor = modeloAdaBoost

resultado = vencedor.predict(validacao_dados)

acertos = (resultado==validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
print(msg)




acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d" % len(validacao_dados))


