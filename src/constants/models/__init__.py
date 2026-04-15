from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


MODELS = {
    "Random_Forest_Regressor": RandomForestRegressor(verbose=1,n_jobs=-1),
    "GradientBoost_Regressor": GradientBoostingRegressor(verbose=1),
    "XGBoost_Regressor": XGBRegressor()
}


PARAMETERS = {
    "Random_Forest_Regressor":{
        #     "max_depth": [5, 8, 15, None, 10],
        #     "max_features": [5, 7, "auto", 8],
        #     "min_samples_split": [2, 8, 15, 20],
        #     "n_estimators": [100, 200, 500, 1000]
    },
    "GradientBoost_Regressor":{
        #     "loss": ['squared_error','huber','absolute_error'],
        #     "criterion": ['friedman_mse','squared_error','mse'],
        #     "min_samples_split": [2, 8, 15, 20],
        #     "n_estimators": [100, 200, 500],
        #     "max_depth": [5, 8, 15, None, 10],
        #     "learning_rate": [0.1, 0.01, 0.02, 0.03]
    },
    "XGBoost_Regressor":{
        #     "learning_rate": [0.1, 0.01],
        #     "max_depth": [5, 8, 12, 20, 30],
        #     "n_estimators": [100, 200, 300, 500, 1000],
        #     "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]
    }
}