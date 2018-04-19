
import pandas as pd
dataset = pd.read_csv('petr4.csv')
#print( dataset.head() )

dataset['Date'] = pd.to_datetime(dataset['Date'])

#print( dataset.head() )
#print( dataset.describe() )

dataset['Variation'] = dataset['Close'].sub(dataset['Open'])

print( dataset.head() )



from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure, Layout, Candlestick


#x1=dataset.Date
#y1=dataset.Close
#data = [Scatter(x=x1, y=y1)]
#layout = Layout(
#   xaxis=dict(
#       range=['01-01-2010','11-04-2017'],
#       title='Ano'
#   ),
#   yaxis=dict(
#       range=[min(x1), max(y1)],
#       title='Valor da Acao'
#   ))

#fig = Figure(data = data, layout = layout)

#plot(fig)


# candles dos ultimos 6 meses..
#dataset2 = dataset.head(180)
#dados = Candlestick(x=dataset2.Date,
#                       open=dataset2.Open,
#                       high=dataset2.High,
#                       low=dataset2.Low,
#                       close=dataset2.Close,
#                       )
#data=[dados]
##py.offline.iplot(data,filename='grafico_candlestick')
#plot(data,filename='grafico_candlestick.html')
