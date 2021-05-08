# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:34:41 2021

@author: dsouzm3
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

# df = pd.read_csv(r"C:\Users\dsouzm3\Desktop\Melwyn\Online\train.csv")
df = pd.read_csv(r"C:\Users\dsouzm3\Desktop\Machine Learning Self Scripts\House price prediction\train.csv")

# print(df.head(100))
# print(df.info())

# # df_test = pd.read_csv(r"C:\Users\dsouzm3\Desktop\Melwyn\Online\test.csv")

df['POSTED_BY'].replace({'Dealer':0,'Owner':1,'Builder':2},inplace=True)
df['BHK_OR_RK'].replace({'BHK':0,'RK':1},inplace=True)

# print(df.head())


x = df[["POSTED_BY", "UNDER_CONSTRUCTION","RERA","BHK_NO.","BHK_OR_RK","SQUARE_FT","READY_TO_MOVE","RESALE","LONGITUDE","LATITUDE"]]
y = df[["TARGET(PRICE_IN_LACS)"]]
# # print(df.POSTED_BY.value_counts())

df.hist(bins=100, figsize=(35, 35))
plt.show()

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=False)

# scaler = preprocessing.MinMaxScaler()
# x_train = scaler.fit_transform(x_train)

# print(type(x_train))
# print(type(y_train))
# # input()


# model = LogisticRegression()
# model.fit(x_train,y_train.to_numpy())

# x_test = scaler.transform(x_test)
# y_hats = model.predict(x_test)

# y_test = pd.DataFrame(y_test)
# y_hats = pd.DataFrame(y_hats)
# # print(y_hats.info())
# y_hats.rename(columns = {0:"Predicted_values"}, inplace = True)

# df1 = pd.concat([y_test.reset_index(drop='Tru‌​e'),y_hats.reset_index(drop='Tru‌​e')],axis=1)

# print(df1.head(100))
# input()



