import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


from sklearn.feature_selection import f_regression, mutual_info_regression
import pandas as pd

def correlation_selected_feature(df, quantile):

    """
    :param df: a dataframe, with both y and X
    :param quantile: quantile for select feature, i.e. mid, quarter ..., give an integer, mid == 2, quarter == 4, etc.
    :return: The original dataframe with only selected features
    """

    correlations = df.corr().iloc[0,:][1:]
    correlation_threshold = (abs(correlations).max() - abs(correlations).min()) / quantile
    selected_feature = correlations[abs(correlations) >= correlation_threshold].sort_values().index

    return df[selected_feature]

def F_selected_feature(df, F_threshold):
    """
    :param df: a dataframe, with both y and X
    :param F_threshold: select how many features (depend on f_score, the higher the better)
    :return: The original dataframe with only selected features
    """
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    f_score, pvalues = f_regression(X, y)
    f = pd.DataFrame(f_score, index=df.columns[1:]).iloc[:,0].sort_values(ascending=False)
    selected_feature = f[:F_threshold].index

    return df[selected_feature]


def mutual_info_selected_feature(df, MI_threshold, N=5):
    """
    :param df: a dataframe, with both y and X
    :param N: number of neighbors for calculating the mutual info value, default = 5
    :param MI_threshold: select how many features (depend on MI_score, the higher the better)
    :return: The original dataframe with only selected features
    """
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    MI_score = mutual_info_regression(X, y)
    MI = pd.DataFrame(MI_score, index=df.columns[1:]).iloc[:,0].sort_values(ascending=False)
    selected_feature = MI[:MI_threshold].index

    return df[selected_feature]

def RMSLE(y_test, y_pred):
    """
    :param y_test: actual series
    :param y_pred: predicted series
    :return: calculated metric
    """
    N = len(y_test)
    log_pred = np.log(y_pred + np.ones(N))
    log_test = np.log(y_test + np.ones(N))
    sum_square = sum((log_pred - log_test)**2)
    rmsle = (sum_square * (1/N)) ** 0.5
    return rmsle

def OLS_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        "fit_intercept": [True, False],
    }

    OLS_model_GS = GridSearchCV(
        LinearRegression(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {OLS_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{OLS_model_GS.best_score_:.3f}')
    print('-----')

    OLS_model_GS_pred = OLS_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameters' : OLS_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, OLS_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, OLS_model_GS_pred),
        'RMSLE' : RMSLE(y_test, OLS_model_GS_pred)
    }
    return metric_dict

def Ridge_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        "alpha": np.linspace(1e-14, 1, 50),
        "fit_intercept": [True, False],
        "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

    Ridge_model_GS = GridSearchCV(
        Ridge(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {Ridge_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{Ridge_model_GS.best_score_:.3f}')
    print('-----')

    Ridge_model_GS_pred = Ridge_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameters' : Ridge_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, Ridge_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, Ridge_model_GS_pred),
        'RMSLE' : RMSLE(y_test, Ridge_model_GS_pred)
    }
    return metric_dict

def Lasso_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        "alpha": np.linspace(1e-14, 1, 50),
        "fit_intercept": [True, False],
    }

    Lasso_model_GS = GridSearchCV(
        Lasso(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {Lasso_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{Lasso_model_GS.best_score_:.3f}')
    print('-----')

    Lasso_model_GS_pred = Lasso_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameter' : Lasso_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, Lasso_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, Lasso_model_GS_pred),
        'RMSLE' : RMSLE(y_test, Lasso_model_GS_pred)
    }
    return metric_dict

def eNet_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        "max_iter": [1, 5, 10, 15, 20, 25, 30],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": np.arange(0.0, 1.0, 0.1)
    }

    eNet_model_GS = GridSearchCV(
        ElasticNet(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {eNet_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{eNet_model_GS.best_score_:.3f}')
    print('-----')

    eNet_model_GS_pred = eNet_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameter' : eNet_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, eNet_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, eNet_model_GS_pred),
        'RMSLE' : RMSLE(y_test, eNet_model_GS_pred)
    }
    return metric_dict


def KNN_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        'n_neighbors': range(1,50),
        'weights': ['uniform', 'distance'],
    }

    knn_model_GS = GridSearchCV(
        KNeighborsRegressor(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {knn_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{knn_model_GS.best_score_:.3f}')
    print('-----')

    knn_model_GS_pred = knn_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameter' : knn_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, knn_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, knn_model_GS_pred),
        'RMSLE' : RMSLE(y_test, knn_model_GS_pred)
    }
    return metric_dict


def decision_tree_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        "max_depth": range(1,50),
        "min_samples_split": range(1,15),
        "min_samples_leaf": range(1,20)
    }

    dt_model_GS = GridSearchCV(
        DecisionTreeRegressor(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {dt_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{dt_model_GS.best_score_:.3f}')
    print('-----')

    dt_model_GS_pred = dt_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameter' : dt_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, dt_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, dt_model_GS_pred),
        'RMSLE' : RMSLE(y_test, dt_model_GS_pred)
    }
    return metric_dict

def GBDT_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        'learning_rate': [0.01,0.02,0.03,0.04],
        'subsample'    : [0.9, 0.5, 0.2, 0.1],
        'n_estimators' : [100,500,1000, 1500],
        'max_depth'    : [4,6,8,10]
    }

    gbdt_model_GS = GridSearchCV(
        GradientBoostingRegressor(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {gbdt_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{gbdt_model_GS.best_score_:.3f}')
    print('-----')

    gbdt_model_GS_pred = gbdt_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameter' : gbdt_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, gbdt_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, gbdt_model_GS_pred),
        'RMSLE' : RMSLE(y_test, gbdt_model_GS_pred)
    }
    return metric_dict

def xgb_feature_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    xgb_model_GS = GridSearchCV(
        xgb.XGBRegressor(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {xgb_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{xgb_model_GS.best_score_:.3f}')
    print('-----')

    xgb_model_GS_pred = xgb_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameter' : xgb_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, xgb_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, xgb_model_GS_pred),
        'RMSLE' : RMSLE(y_test, xgb_model_GS_pred)
    }
    return metric_dict

def random_forest_CV(X_train_valid, y_train_valid, X_test, y_test):

    param = {
        'bootstrap': [True, False],
#         'max_depth': [10, 30, 50, 70, 90, None],
        'max_depth': [None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
#         'min_samples_split': [2, 5, 10],
        'min_samples_split': [2],
        'n_estimators': [200, 400, 600]
    }

    rf_model_GS = GridSearchCV(
        RandomForestRegressor(),
        param,
        cv=10,
        verbose=2,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    ).fit(X_train_valid, y_train_valid)
    print('-----')
    print(f'Best parameters {rf_model_GS.best_params_}')
    print(
    f'Mean cross-validated accuracy score of the best_estimator: ' + \
    f'{rf_model_GS.best_score_:.3f}')
    print('-----')

    rf_model_GS_pred = rf_model_GS.best_estimator_.predict(X_test)
    metric_dict = {
        'Best parameter' : rf_model_GS.best_params_,
        'Mean squared error' : mean_squared_error(y_test, rf_model_GS_pred),
        'Coefficient of determination' : r2_score(y_test, rf_model_GS_pred),
        'RMSLE' : RMSLE(y_test, rf_model_GS_pred)
    }
    return metric_dict














