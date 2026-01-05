import sys
import os

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    '''
    This function is responsible for saving the object to a file using pickle.
    '''
    try:
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
            logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    """
    Trains models using GridSearchCV and returns report
    """
    try:
        report = {}

        for model_name, model in models.items():

            logging.info(f"Training started for {model_name}")

        
            # 🚫 CATBOOST SKIP FROM GRIDSEARCH
            
            if model_name == "CatBoost Regressor":
                logging.info("Skipping GridSearchCV for CatBoost")

                model.fit(x_train, y_train)
                y_test_pred = model.predict(x_test)
                score = r2_score(y_test, y_test_pred)

                report[model_name] = {
                    "best_model": model,
                    "best_score": score,
                    "best_params": "default"
                }
                continue   # 👈 VERY IMPORTANT

            # ✅ GridSearchCV for all other models

            # 🔹 Get params for this model
            param_grid = params.get(model_name, {})

            # 🔹 GridSearchCV
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,  ### → 3-fold cross validation
                        ###→ data 3 parts me banta hai
                scoring="r2",
                n_jobs=-1 ##CPU ke saare cores use karo (fast)
            )

            # 🔹 Train with tuning
            gs.fit(x_train, y_train)

            # 🔹 Best tuned model
            best_model = gs.best_estimator_

            # 🔹 Prediction
            y_test_pred = best_model.predict(x_test)

            # 🔹 Score
            score = r2_score(y_test, y_test_pred)

            # 🔹 Final report entry
            report[model_name] = {
                "best_model": best_model,
                "best_score": score,
                "best_params": gs.best_params_
            }

            logging.info(
                f"{model_name} | R2: {score} | Params: {gs.best_params_}"
            )

        return report


    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException(e, sys)