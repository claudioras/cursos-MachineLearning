
from collections import Counter

import pandas as pd
data_frame = pd.read_csv('candles.csv')
data_frame['Variacao'] = data_frame['Maxima'].sub(data_frame['Minima'])

#AltaBaixa,Abertura,Fechamento,Maxima,Minima,FechouNaMaxima,FechouNaMinima,AbriuNaMaxima,AbriuNaMinima,Martelo,Martelo_Invertido,MM17Tocou,MM34Tocou,AcimaMM17,AbaixoMM17,VolumeAcimaMedia,FF_FD,TocouBB_Inferior,TocouBB_Superior,IFR_Menor30,IFR_Maior70
#X_df=data_frame[['AltaBaixa','FechouNaMaxima','FechouNaMinima','Martelo','Martelo_Invertido','VolumeAcimaMedia','FF_FD','TocouBB_Inferior','TocouBB_Superior','IFR_Menor30','IFR_Maior70']]
X_df=data_frame[['AltaBaixa','FechouNaMaxima','FechouNaMinima','AbriuNaMaxima','AbriuNaMinima','Martelo','Martelo_Invertido','MM17Tocou','MM34Tocou','AcimaMM17','AbaixoMM17','VolumeAcimaMedia','FF_FD','TocouBB_Inferior','TocouBB_Superior','IFR_Menor30','IFR_Maior70']]
Y_df=data_frame['Proximo']
#Y_df=data_frame['Variacao']

DataHora=data_frame['DataHora']




print( data_frame.head() )


#pega os dummies da coluna alfanumerica 'AltaBaixa'
Xdummies_df = pd.get_dummies(X_df)

#Quando o array jah eh binario ao rodar o metodo get_dummies ele retorna duas colunas onde a segunda coluna eh o valor original, logo outra abordagem eh: Ydummies = pd.get_dummies(Y)[1], onde 1 eh o titulo da coluna
#Ydummies_df = pd.get_dummies(Y_df)
Ydummies_df = Y_df




X = Xdummies_df.values
Y = Ydummies_df.values




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

validacao_DataHora = DataHora[fim_de_teste:]


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):

    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    acertos = (resultado==teste_marcacoes)
    #acertos = []


#    ind=0
#    for tm in teste_marcacoes:
#        diferencas = resultado[ind] - tm
#
#
#        erro = False
#        for item in diferencas:
#            if item <> 0:
#               erro = True
#        if erro==True:
#           acertos.append(False)
#        else:
#           acertos.append(True)
#
#        ind += 1

      
        


    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do {0}: {1}".format(nome,taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaboostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

maximo = max(resultados)
vencedor = resultados[maximo]



resultado = vencedor.predict(validacao_dados)

acertos = (resultado==validacao_marcacoes)

#validacao_dados = X[fim_de_teste:]
#validacao_marcacoes = Y[fim_de_teste:]
#validacao_DataHora = DataHora[fim_de_teste:]


print("")
print(DataHora[len(DataHora)-2])
print(validacao_dados[len(validacao_dados)-2])
#X_df=data_frame#[['AltaBaixa','FechouNaMaxima','FechouNaMinima','Martelo','Martelo_Invertido','VolumeAcimaMedia','FF_FD','TocouBB_Inferior','TocouBB_Superior','IFR_Menor30','IFR_Maior70']]

print("Prediccao para o proximo candle: {}".format(resultado[len(resultado)-2]))

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto do vencedor entre todos os algoritmos no mundo real: {0}".format(taxa_de_acerto)
print(msg)




acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d" % len(validacao_dados))


