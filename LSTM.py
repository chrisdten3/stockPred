import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, LSTM

st.title('Stock Future Forcaster')
st.subheader('This is a stock forecaster made by Chris Tengey (cdt50@georgetown.edu), a Sophomore at Georgetown University. This forecaster uses the Yahoo Finance packgage to access historical stock data including the daily opening price of a stock. A Long Short Term Memory (LSTM) Recurrent Neural Netowrk is then built. The neural network predicts through fundamental analysis, where only the quantitative aspects of a stock are considered.')

st.write('Here is a list of stock tickers avaliable on yahoo finance https://finance.yahoo.com/lookup/')
symbol = st.text_input('Your Selectd Stock is...').upper()

if symbol is None:
    st.write('Once you have entered a stock, the program will start running')

else:
    data = yf.download(tickers=symbol, period='5y', interval='1d')
    opn = data['Open']
    ds = opn.values

    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

    #define test and training data size
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size

    #splittig data between train and test set
    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #using 100 previous days as a period for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)

    #reshaping data to fit in LSTM model
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #creating LSTM model using keras
    model = Sequential() # A sequential model allows for a nueral network to be built layer by layer
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    #first layer
    #50 cells are added to the layer
    #because return_sequence is true, the LSTM layer will return the entire sequence rather than just the last output
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))
    #This is the output layer, it has one cell and a linear acitvation functioj
    #the output will then be some continious value
    model.summary()

    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)

    #predicting on train and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    #inverse transform to get actual value, basically un normalizing the data
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    gen_ds_test = len(ds_test) - 100
    fut_inp = ds_test[gen_ds_test:]

    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    fut_inp.shape
    tmp_inp = tmp_inp[0].tolist()


    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    #creating a dummy plane to plot the predicted on top of the actual
    plot_new=np.arange(1,101)
    plot_pred=np.arange(101,131)
    lastHun = len(ds_scaled) - 100
    ds_new = ds_scaled.tolist()

    #creating final data for plotting
    final_graph = normalizer.inverse_transform(ds_new).tolist()
    #Plotting final results with predicted value after 30 Days

    fig = plt.figure()

    plt.plot(final_graph,)
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.title("{0} prediction of next month open".format(symbol))
    plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
    plt.legend()

    st.pyplot(fig)

