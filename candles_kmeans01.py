
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb



data_frame = pd.read_csv('candles.csv')


#AltaBaixa,Abertura,Fechamento,Maxima,Minima,FechouNaMaxima,FechouNaMinima,AbriuNaMaxima,AbriuNaMinima,Martelo,Martelo_Invertido,MM17Tocou,MM34Tocou,AcimaMM17,AbaixoMM17,VolumeAcimaMedia,FF_FD,TocouBB_Inferior,TocouBB_Superior,IFR_Menor30,IFR_Maior70
#X_df=data_frame[['AltaBaixa','FechouNaMaxima','FechouNaMinima','Martelo','Martelo_Invertido','VolumeAcimaMedia','FF_FD','TocouBB_Inferior','TocouBB_Superior','IFR_Menor30','IFR_Maior70']]
#AltaBaixa,Horario,GrandePequeno,PercMaximaVsFechamento,PercMinimaVsFechamento,PercFechamentoVsFechamento,Target


X_df=data_frame[['AltaBaixa','GrandePequeno','PercMaximaVsFechamento','PercMinimaVsFechamento','PercFechamentoVsFechamento','Target']]

X_df = X_df.drop('Target', axis = 1)

X = pd.get_dummies(X_df)




#sb.pairplot(X_df,hue='Target')
sb.pairplot(X)
pl.show()


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

X['k-classes'] = kmeans.labels_

print(X)

