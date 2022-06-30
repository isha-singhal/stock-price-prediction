from urllib.parse import uses_relative
from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from math import sqrt

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Price Prediction using LSTM')
st.subheader('Visualizations')
#dow = si.tickers_dow()
user_input = st.selectbox(
        'Select Stock Ticker',
     ('AAPL', 'TSLA', 'MSFT','CCL','AMZN','BTC-USD'))
df = data.DataReader(user_input, 'yahoo',start,end)

#Describing Data

st.subheader('Data from {} to {}'.format(start,end))
st.write(df.describe())

#Visualizations
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,8))
plt.plot(df['Close'])
plt.xlabel('Time')
plt.ylabel('Closing Price')
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma100,label='100MA')
plt.plot(df['Close'],label='Closing Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Closing Price')
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma100,'r',label='100MA')
plt.plot(ma200,'g',label='200MA')
plt.plot(df['Close'],'b',label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

#splitting data into training and testing
df_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
df_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


#scaling of data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

df_train_array = scaler.fit_transform(df_train)

#Load my Model
model = load_model('keras_model.h5')

#Testing part
past_100_days = df_train.tail(100)
final_df = past_100_days.append(df_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

#Making Predictions
y_predicted = model.predict(X_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12,8))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Mean Squared Error, Mean absolute error, Root Mean Squared Error, R-Squared Error
mae1=mae(y_test,y_predicted)
#print('Mean absolute error is:',mae1)
mse1=mse(y_test,y_predicted)
#print('Mean Squared Error',mse1)
rmse1=sqrt(mse1)
#print('Root Mean Squared Error',rmse1)
r2_score=r2s(y_test,y_predicted)
#print('R-Squared Error',r2_score)

st.subheader('Accuracy of this model is:')
st.write(r2_score*100)
