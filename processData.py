#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
reload(sys)
sys.setdefaultencoding('utf-8')
import overlook
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def countDifferentCities(df):
    return df['City'].value_counts().shape[0]

def tableDifferentCities(df):
    absolute = df['City'].value_counts()
    percentage = df['City'].value_counts()/df['City'].value_counts().sum()
    rdf = pd.concat([absolute, percentage], axis=1)
    rdf.columns = ['Abs', 'Frac']
    return rdf

def computeDaysSinceOpen(df):
    lastDate = pd.to_datetime('01/01/2015')

    df['Days'] = lastDate-df['Open Date']

    df['Days'] = df['Days'].astype('timedelta64[D]').astype(int)

def computeOpeningMonth(df):
    lastDate = pd.to_datetime('01/01/2015')
    df['Month'] = df['Open Date'].map(lambda d: int((lastDate-d).days)/30)
    df['Month'][df['Days'] > 18*30] = 0 #add this info to improve linear regression
    df['Month'] = df['Month'].astype('timedelta64[D]').astype(int)

def transformType(df):
    def MBtoDT(t):
        if t == 'MB': return 'DT'
        else: return t
    df['Type'] = df['Type'].map(MBtoDT)

def getCitiesWithMoreData(df, counts=5):
    c = df['City'].value_counts()
    return c[c > counts].index.values

def assignCities(df, df_train, mincounts):
    cities = getCitiesWithMoreData(df_train, mincounts) # with more data compared to train
    b = df['City'].isin(cities)
    df['City'][~df['City'].isin(cities)] = 'other'

def transformPs(df_train, allData):
    ps = ['P%d'%n for n in range(1,38)]
    skewed = df_train[ps].skew()
    skewed = skewed[skewed > 0.5].index
    allData[skewed] = np.log1p(allData[skewed])

def main():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")


    print "training set diff cities", countDifferentCities(df_train)
    #print "test set diff cities", countDifferentCities(df_test)

    #trainDiff = tableDifferentCities(df_train)
    #testDiff = tableDifferentCities(df_test)

    y = np.log1p(df_train['revenue'])

    allData = pd.concat([df_train, df_test])
    allData.drop('revenue', axis=1, inplace=True)

    allData['Open Date'] = pd.to_datetime(allData['Open Date'])

    computeDaysSinceOpen(allData)
    computeOpeningMonth(allData)
    allData['Days'] = np.log1p(allData['Days'])
    allData.drop('Open Date', axis=1, inplace=True)

    """
    from sklearn.preprocessing import StandardScaler
    stdSc = StandardScaler()
    X_train.loc[:, quantitative] = stdSc.fit_transform(X_train.loc[:, quantitative])
    X_test.loc[:, quantitative] = stdSc.transform(X_test.loc[:, quantitative])
    """

    transformType(allData)

    assignCities(allData, df_train, 4)

    transformPs(df_train, allData)

    allData = pd.get_dummies(allData)

    allData[:137].to_csv("train_clean.csv", index_label=False)
    allData[137:].to_csv("test_clean.csv", index_label=False)
    df_y = pd.DataFrame({'Id':range(1,138), 'revenue':y.values})
    df_y.to_csv("revenue_clean.csv", index_label=False)


if __name__ == "__main__":
    main()
