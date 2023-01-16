#Local URL: http://localhost:8501
#Network URL: http://192.168.86.59:8501

# Import necessary modules
import streamlit as st
import io
from io import StringIO


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import datetime as dt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import plotly.express as px
from plotly import graph_objs as go
from sklearn.metrics import r2_score


# st.title("STOCK MARKET PREDICTIONS")

st.set_page_config(
    page_title="STOCK MARKET PREDICTIONS",
    page_icon="🎯",
    layout="wide",
)

st.markdown("<h1 style='text-align: center; color: white;'>STOCK MARKET PREDICTIONS</h1>", unsafe_allow_html=True)

st.markdown('##')
st.markdown('##')
st.markdown('##')


with st.container():
   st.header("WHAT IS STOCK?", anchor=None)
   st.write("💰A stock market is a public market where you can buy and sell shares for publicly listed companies.")
   st.write("💰The stocks, also known as equities, represent ownership in the company.")
   st.write("💰The stock exchange is the mediator that allows the buying and selling of shares.")

st.markdown('##')


with st.container():
   st.header("IMPORTANCE OF STOCK", anchor=None)
   st.write("💰Stock markets help companies to raise capital.")
   st.write("💰It helps generate personal wealth.")
   st.write("💰Stock markets serve as an indicator of the state of the economy.")
   st.write("💰It is a widely used source for people to invest money in companies with high growth potential.")

st.markdown('##')

with st.container():
   st.header("STOCK PREDICTION", anchor=None)
   st.write("💰In the finance world stock trading is one of the most important activities.")
   st.write("💰Stock market prediction is an act of trying to determine the future value of a stock other financial instrument traded on a financial exchange.")
   st.write("💰The accurate prediction of share price movement will lead to more profit investors can make.")
   st.write("💰Stock markets are known for their volatility, dynamics, and nonlinearity.")
   st.write("💰With multiple factors involved in predicting stock prices, it is challenging to predict stock prices with high accuracy, and this is where machine learning plays a vital role.")
   st.write("💰Time series forecasting (predicting future values based on historical values) applies well to stock forecasting.")

st.markdown('##')

with st.container():
   st.header("BENEFITS OF STOCK PREDICTION", anchor=None)
   st.write("💰Remove the investment bias")
   st.write("💰Develops the habit of complete analysis")
   st.write("💰Minimizes your losses")
   st.write("💰Assures consistency")
   st.write("💰Gives a better idea about Entry and Exit points")
   st.write("💰Allows the smart way of making money")

st.markdown('##')

with st.container():
    st.header("OBJECTIVE OF THIS PROJECT:", anchor=None)
    st.write("🚩This project aims to discover the future value of a company stock and other financial assets traded on an exchange.")
    st.write("🚩It facilitates a proper time frame to invest in the targeted stock and, forecasting any market crashes that might be imminent in the near future.")
    st.write("🚩The machine learning model assigns weights to each market feature and determines how much history the model should look at to predict future stock prices.")

st.markdown('##')

with st.container():
    st.header("DATASET FOR THIS PROJECT:", anchor=None)
    st.write("📑Stock data is retrieved from Yahoo Finance, which provides free stock quotes.") 
    st.write("📑The model is compatible with all company datasets. NETFLIX.Inc company was selected for this demo session.") 
    st.write("📑Training set : Netflix stock price from 2016 to 2021") 
    st.write("📑Training set : Netflix stock price from Jan 2022 to Oct 2022") 
    st.write("📑The task is to predict the trend of the stock price for 2022-2023.")

st.markdown('##')


dataset_train = pd.read_csv("NFLX_train.csv")

training_set = dataset_train.iloc[:, 1: 2].values

x_dates =  [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dataset_train['Date']]

st.markdown('##')
st.markdown('##')
st.markdown('##')


fig = px.line(x=x_dates, y=dataset_train['Open'], labels={"x":"Time Scale", "y":"Currency in USD "},title='Real Stock Price of NETFLIX.INC',width=1400,height=700)
fig.update_traces(line_color='#00D7FF')
st.plotly_chart(fig)

with st.expander("Explanation"):
    st.write("👉🏻The chart above shows the real stock price of NETFLIX.Inc from Jan 01,2016 to Dec 31,2021.")
    st.write("👉🏻This graph indicates that NETFLIX.Inc stock prices have increased progressively from 109 USD to 612.98 USD.")


sc = StandardScaler()

training_set_scaled = sc.fit_transform(training_set)


X_train = []
y_train = []
n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 5 # Number of past days we want to use to predict the future.



for i in range(n_past, len(training_set_scaled)- n_future +1):
    X_train.append(training_set_scaled[i-n_past: i, 0:training_set_scaled.shape[1]])
    y_train.append(training_set_scaled[i+ n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train) 

X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
# add 1st lstm layer
model.add(LSTM(units = 75, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(rate = 0.2))

# add 2nd lstm layer: 32 neurons
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

model.add(LSTM(units = 32, return_sequences = False))
model.add(Dropout(rate = 0.2))

# add output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = model.fit(X_train, y_train,  epochs=1, batch_size=32, verbose=1)

# fig = px.line(y=history.history['loss'], labels={"x":"epoch", "y":"loss "},title='model loss',width=1400,height=600)
# fig.update_traces(line_color='#82CD47')
# st.plotly_chart(fig)



dataset_test = pd.read_csv("NFLX_test.csv")

real_stock_price = dataset_test.iloc[:, 1: 2].values

    #vertical concat use 0, horizontal uses 1
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), 
                        axis = 0)
#use .values to make numpy array
inputs = dataset_total[len(dataset_total) - len(dataset_test) - n_past:].values

#reshape data to only have 1 col
inputs = inputs.reshape(-1, 1)

#scale input
inputs = sc.transform(inputs)

X_test = []
for i in range(n_past, len(inputs)):
    X_test.append(inputs[i-n_past:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)

#inverse the scaled value
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

data = {"Real_Price":real_stock_price[:,0],"Predicted_Price":predicted_stock_price[:,0]}

df = pd.DataFrame(data,index=dataset_test['Date'])

st.markdown('##')
st.markdown('##')
st.markdown('##')


fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Real_Price"], name="Real Stock Price",line_color='#FF1700'))
fig.add_trace(go.Scatter(x=df.index, y=df["Predicted_Price"], name="Predicted Stock Price",line_color='#06FF00'))
fig.layout.update(title_text='Real Stock Price Vs Predicted Stock Price of NETFLIX.INC', xaxis_rangeslider_visible=True,width=1400,height=850)
st.plotly_chart(fig)

r2_score(real_stock_price, predicted_stock_price)

train_dates = pd.to_datetime(dataset_test['Date'])

n_future = 200
forecast_period_dates = pd.date_range(list(train_dates)[-1],periods=n_future,freq='1d').tolist()
forecast = model.predict(X_train[-n_future:])

forecast_copies = np.repeat(forecast, dataset_test.shape[1], axis=-1)
y_pred_future = sc.inverse_transform(forecast_copies)[:,0]

forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

original = dataset_test[['Date', 'Open']]

df_forecast = pd.DataFrame({'Forecast_Date':np.array(forecast_dates), 'Forecast_Price':y_pred_future})

df_forecast["Forecast_Price"].iloc[0]=original["Open"].iloc[-1]

st.markdown('##')
st.markdown('##')
st.markdown('##')


fig = px.line(df_forecast[1:],x='Forecast_Date', y='Forecast_Price',labels={"x":"Time Scale", "y":"Currency in USD "},title='Forecasted Stock Price of NETFLIX.INC',height=700,width=1400)
fig.update_traces(line_color='#FFE400')
st.plotly_chart(fig)

st.write(df_forecast[df_forecast.Forecast_Price == df_forecast.Forecast_Price.max()]) #Maximum expected stock price value of Netflix