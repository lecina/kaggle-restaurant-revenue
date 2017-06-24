import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.linear_model import LinearRegression, ElasticNet
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn import ensemble, tree, linear_model

def rmse_cv(model, X_train, y, s=""):
    print "running model %s" %s
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)

def rmse_cv_wrapper(model, X_train, y):
    return -rmse_cv(model, X_train, y).mean()
    
def predictWithGrid(X, y, Xt, model, parameter_grid, cv=3):
    grid_search = GridSearchCV(model,
                               scoring="neg_mean_squared_error",
                               param_grid=parameter_grid, 
                               cv=cv)
    grid_search.fit(X, y)
    model = grid_search.best_estimator_
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    rmse = rmse_cv(model, X, y, "grid")
    print "Error: ", rmse.mean()
    model.fit(X,y)
    yp=model.predict(Xt)
    return yp, model

def predictWithRandomForest(X, y, Xt):
    #deprecated
    parameter_grid = {
        'n_estimators' : [6,7,8,9],
        'max_features' : ['sqrt'],
        'max_depth' : [3,4],
        'min_samples_leaf' : [1,2]
    }
    model=RandomForestRegressor()
    grid_search = GridSearchCV(model,
                               scoring="neg_mean_squared_error",
                               param_grid=parameter_grid,
                               cv=3)
    grid_search.fit(X, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    rmse = rmse_cv(model, X, y, "random forest")
    print "Error: ", rmse.mean()
    model.fit(X,y)
    yp=model.predict(Xt)
    return yp

def predictWithKNN(X, y, Xt):
    #deprecated
    parameter_grid = {
        'n_neighbors' : [14],
        'weights' : ['uniform', 'distance']
    }
    model=KNeighborsRegressor()
    grid_search = GridSearchCV(model,
                               scoring="neg_mean_squared_error",
                               param_grid=parameter_grid,
                               cv=3)
    grid_search.fit(X, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    rmse = rmse_cv(model, X, y, "random forest")
    print "Error: ", rmse.mean()
    model.fit(X,y)
    yp=model.predict(Xt)
    return yp

def predicLasso(X, y, Xt):
    model_lasso = LassoCV(alphas = [0.06]).fit(X, y)
    coef = pd.Series(model_lasso.coef_, index = X.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    print "Lasso", rmse_cv(model_lasso, X, y, "lasso").mean()
    model_lasso.fit(X, y)
    alpha = model_lasso.alpha_
    print("Best alpha :", alpha)
    rmse = rmse_cv(model_lasso, X, y, "lasso")
    print "Error: ", rmse.mean()
    return model_lasso.predict(Xt)

def predictEN(X, y, Xt):
    #equiv to Ridge, as best l1_ratio=0.0
    alphas = [0.00, 0.25, 0.5, 0.75, 1.0]
    cv_EN = [rmse_cv(ElasticNet(alpha = 0.04, l1_ratio=alpha, max_iter=5000), X, y).mean() for alpha in alphas]
    cv_EN = pd.Series(cv_EN, index = alphas)
    print cv_EN
    print "EN", cv_EN.min()
    modelEN = ElasticNet(alpha=cv_EN.min())
    modelEN.fit(X, y)
    cv_EN.plot(title = "Validation")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.show()
    return modelEN.predict(Xt)

def predictRidge(X,y,Xt):
    #deprecated
    model_ridge = Ridge()
    alphas = [3,4,4.5,5,5.5,6,10]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha), X,y, "ridge").mean() 
                for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index = alphas)
    alpha = cv_ridge.idxmin()
    print "Best ridge alpha", alpha
    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X, y)

    cv_ridge.plot(title = "Validation")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.show()
    return model_ridge.predict(Xt)

def predict(model, X, y, Xt, s=""):
    rmse = rmse_cv(model, X, y, s)
    print "Error %s: "%s, rmse.mean()
    model.fit(X,y)
    return model.predict(Xt), model

def plot_model_var_imp( model , X , y ):
    try:
        imp = pd.DataFrame( 
            model.coef_  , 
            columns = [ 'Importance' ] , 
            index = X.columns 
        )
    except:
        imp = pd.DataFrame( 
            model.feature_importances_  , 
            columns = [ 'Importance' ] , 
            index = X.columns 
        )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp.plot( kind = 'barh' )
    return imp

def plotAfter(model, modelAfter, df_train, df_train2, df_test, df_test2):
    modelRidgeAfter = Ridge(alpha=10)
    predictionRidgeAfter, modelRidgeAfter = predict(modelRidgeAfter, df_train2, y, df_test2, s="ridge")

    preds = pd.DataFrame({"preds":modelRidge.predict(df_train), "true":y, "predsAfter":modelRidgeAfter.predict(df_train2)})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds["residualsAfter"] = preds["true"] - preds["predsAfter"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter")
    plt.scatter(preds['preds'], preds['residuals'], c = "blue", marker = "s", label = "Before")
    plt.scatter(preds['predsAfter'], preds['residualsAfter'], c = "red", marker = "v", label = "After")
    plt.show()

def main():
    df_train = pd.read_csv("train_clean.csv")
    df_test = pd.read_csv("test_clean.csv")
    y = pd.read_csv("revenue_clean.csv")

    df_train = df_train.set_index('Id')
    df_test = df_test.set_index('Id')
    y = y['revenue']


    #df_train2 = df_train.drop(['P24','Month','P9','P1','P27','P2','City_other','P36','P16','P31','P11','P4','City_Bursa','P12','P14','Type_DT'], axis=1)
    #df_test2 = df_test.drop(['P24','Month','P9','P1','P27','P2','City_other','P36','P16','P31','P11','P4','City_Bursa','P12','P14','Type_DT'], axis=1)
    #df_train2 = df_train.drop(["P24","Month","P9","P1","P27","P2","City_other","P36","P16","P31","P11","P4","City_Bursa","P12","P14","Type_DT","P13","City_Samsun","P5","P20","P10","City Group_Big Cities","City Group_Other","P19","P33","P29","Type_IL","P25"], axis=1)
    #df_test2 = df_test.drop(["P24","Month","P9","P1","P27","P2","City_other","P36","P16","P31","P11","P4","City_Bursa","P12","P14","Type_DT","P13","City_Samsun","P5","P20","P10","City Group_Big Cities","City Group_Other","P19","P33","P29","Type_IL","P25"], axis=1)

    df_train2 = df_train.drop(["P24","P27","P9","P11","P31","P1","City_Bursa","P12","P7","P14","Type_DT","P16","P4","P36","P2","City_other"], axis=1)
    df_test2 = df_test.drop(["P24","P27","P9","P11","P31","P1","City_Bursa","P12","P7","P14","Type_DT","P16","P4","P36","P2","City_other"], axis=1)
    df_train = df_train2
    df_test = df_test2

#
    """
    parameter_grid = {
        'n_estimators' : [6,7,8,9],
        'max_features' : ['sqrt'],
        'max_depth' : [3,4],
        'min_samples_leaf' : [1,2]
    }
    #modelRFR = RandomForestRegressor(n_estimators=8, max_features='sqrt', max_depth=3, min_samples_leaf=1)
    modelRFR = RandomForestRegressor()
    predictionRFN, modelRFR = predictWithGrid(df_train, y, df_test, modelRFR, parameter_grid)
    #predictionRFN, modelRFR = predict(modelRFR, df_train, y, df_test, s="random forest")
    predictionRFRTrain = modelRFR.predict(df_train)

    parameter_grid = {
        'n_neighbors' : [5,10,15,12,13,14],
        'weights' : ['distance', 'uniform']
    }
    #modelKNN = KNeighborsRegressor(n_neighbors=14, weights='distance')
    modelKNN = KNeighborsRegressor()
    predictionKNN, modelKNN = predictWithGrid(df_train, y, df_test, modelKNN, parameter_grid)
    #predictionKNN, modelKNN = predict(modelKNN, df_train, y, df_test, s="random forest")
    predictionKNNTrain = modelKNN.predict(df_train)
    """

    modelXGB = ensemble.GradientBoostingRegressor() 
    parameter_grid = {'n_estimators' : [30,100,300], 'learning_rate':[0.03,0.1,0.3], 'max_depth' : [3,4,5,6], 'max_features' : ['sqrt'],
                    'min_samples_leaf':[15], 'min_samples_split':[10], 'loss':['huber']}
    predictionXGB, modelXGB = predictWithGrid(df_train, y, df_test, modelXGB, parameter_grid,3)
    predictionXGBTrain = modelXGB.predict(df_train)

    modelXGB2 = ensemble.GradientBoostingRegressor() 
    parameter_grid = {'n_estimators' : [30,100,300], 'learning_rate':[0.03,0.1,0.3], 'max_depth' : [3,4,5,6], 'max_features' : ['sqrt'],
                    'min_samples_leaf':[15], 'min_samples_split':[10], 'loss':['huber']}
    predictionXGB2, modelXGB2 = predictWithGrid(df_train, y, df_test, modelXGB2, parameter_grid, 4)

    modelXGB3 = ensemble.GradientBoostingRegressor() 
    parameter_grid = {'n_estimators' : [30,100,300], 'learning_rate':[0.03,0.1,0.3], 'max_depth' : [3,4,5,6], 'max_features' : ['sqrt'],
                    'min_samples_leaf':[15], 'min_samples_split':[10], 'loss':['huber']}
    predictionXGB3, modelXGB3 = predictWithGrid(df_train, y, df_test, modelXGB3, parameter_grid, 5)
    

    plot_model_var_imp(modelXGB, df_train, y)
    plt.show()


    """
    modelLinear = LinearRegression()
    predictionLinear, modelLinear = predict(modelLinear, df_train, y, df_test, s="linear")
    predictionLinearTrain = modelLinear.predict(df_train)
    """
    df_train = df_train2
    df_test = df_test2
    modelLasso = LassoCV(alphas=[0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.1, 0.3]) #alpha = 0.06
    #parameter_grid = {'alphas':[0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.1, 0.3]}
    predictionLasso, modelLasso = predict(modelLasso, df_train, y, df_test, s="lasso")
    #predictionLasso, modelLasso = predictWithGrid(df_train, y, df_test, modelLasso, parameter_grid)
    predictionLassoTrain = modelLasso.predict(df_train)

    modelRidge = Ridge() #alpha = 6
    parameter_grid = {'alpha' : [4.5, 4.75,5,5.25, 5.5, 5.75,6,7,8,9,10,11,12,13,14,15]}
    predictionRidge, modelRidge = predictWithGrid(df_train, y, df_test, modelRidge, parameter_grid)
    #predictionRidge, modelRidge = predict(modelRidge, df_train, y, df_test, s="ridge")
    predictionRidgeTrain = modelRidge.predict(df_train)



    modelEN = ElasticNet() #(alpha=0.04, l1_ratio=0.001)
    parameter_grid = {'alpha':[0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.1, 0.3], "l1_ratio":[0.0, 0.1, 0.25, 0.5, 0.75, 1]}
    #predictionEN, modelEN = predict(modelEN, df_train, y, df_test, s="EN")
    predictionEN, modelEN = predictWithGrid(df_train, y, df_test, modelEN, parameter_grid)
    predictionENTrain = modelEN.predict(df_train)

    modelEN2 = ElasticNet() #(alpha=0.04, l1_ratio=0.001)
    parameter_grid = {'alpha':[0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.1, 0.3], "l1_ratio":[0.0, 0.1, 0.25, 0.5, 0.75, 1]}
    #predictionEN, modelEN = predict(modelEN, df_train, y, df_test, s="EN")
    predictionEN2, modelEN2 = predictWithGrid(df_train, y, df_test, modelEN2, parameter_grid, cv=4)

    modelEN3 = ElasticNet() #(alpha=0.04, l1_ratio=0.001)
    parameter_grid = {'alpha':[0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.1, 0.3], "l1_ratio":[0.0, 0.1, 0.25, 0.5, 0.75, 1]}
    #predictionEN, modelEN = predict(modelEN, df_train, y, df_test, s="EN")
    predictionEN3, modelEN3 = predictWithGrid(df_train, y, df_test, modelEN3, parameter_grid, cv=5)

    """
    model = modelRidge
    coefs = plot_model_var_imp(model, df_train, y)
    coefs['abs'] = coefs.Importance.map(lambda x: np.abs(x))
    threshold = 0.05
    cols = coefs[coefs['abs'] < threshold].sort_values(['abs'], ascending=True)['abs'].index.values
    print "[\"" + "\",\"".join(cols) + "\"]"
    plt.show()

    #PLOT PREV RESULTS
    plt.scatter(predictionLinearTrain, y-predictionLinearTrain, marker=".", c="g", label="Linear")
    plt.scatter(predictionRidgeTrain, y-predictionRidgeTrain, marker="o", c="b", label="Ridge")
    plt.scatter(predictionLassoTrain, y-predictionLassoTrain, marker="v", c="r", label="Lasso")
    plt.scatter(predictionENTrain, y-predictionENTrain, marker="*", c="b", label="EN")
    plt.xlabel("log(pred)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


    plt.scatter(predictionRFRTrain, y-predictionRFRTrain, marker="v", c="r", label="RFR")
    plt.scatter(predictionENTrain, y-predictionENTrain, marker="*", c="b", label="EN")
    plt.xlabel("log(pred)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()
    """


    #plotAfter(model, modelAfter, df_train, df_train2, df_test, df_test2)

    prediction = np.exp((predictionEN+predictionXGB+predictionEN2+predictionXGB2+predictionEN3+predictionXGB3)/6.)
    df_out = pd.concat([pd.Series(df_test.index.values), pd.Series(prediction)], axis=1, keys=['Id', 'Prediction'])
    df_out.to_csv('submission.csv', index=False)
        

if __name__ == "__main__":
    main()
