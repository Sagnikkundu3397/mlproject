# src/config/model_params.py

MODEL_PARAMS = {
    "Decision Tree": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"]
    },

        "Random Forest": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20]
},

    "Gradient Boosting": {
            "learning_rate": [0.05, 0.1],
            "n_estimators": [50, 100]
},

    "Linear Regression": {},

    "K-Neighbor Regressor": {
            "n_neighbors": [5, 7, 9, 11]
    },

    "XGBRegressor": {
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 200],
            "max_depth": [3, 5]
},


    "AdaBoost Regressor": {
        "learning_rate": [0.1, 0.01, 0.5],
        "n_estimators": [8, 16, 32, 64]
    }
}
