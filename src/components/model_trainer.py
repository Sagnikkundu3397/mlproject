# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
import warnings 
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
import sys
import os
from src.utils import evaluate_models

from src.config.model_params import MODEL_PARAMS

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report: dict = evaluate_models(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, models=models,params=MODEL_PARAMS)
            
            ##to get best model score from list of models and  to get best model name from the dict
            
            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["best_score"]
            )
            
            best_model = model_report[best_model_name]["best_model"]
            best_model_score = model_report[best_model_name]["best_score"]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info(
                f"Best model: {best_model_name} with R2 score: {best_model_score}"
            ) 
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return best_model_name, best_model_score


        except Exception as e:
            logging.error("Error occurred in model trainer")
            raise CustomException(e, sys)   
    
    