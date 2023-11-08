import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import sys
import argparse
sys.path.append('../')

from utils.preprocessing import *
from utils.const import *

def train_tabnet(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv):
    """
    train a tabular neural network model
    """
    model = TabNetRegressor(optimizer_fn=torch.optim.AdamW,
                            optimizer_params=dict(lr=2e-2),
                            scheduler_params={"step_size":50, "gamma":0.5},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR,
                            mask_type='entmax')  # Can be "sparsemax"

    # Fit the model
    model.fit(
        X_train_cv, y_train_cv.reshape(-1, 1),
        eval_set=[(X_eval_cv, y_eval_cv.reshape(-1, 1))],
        eval_name=['test'],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=10,
        batch_size=256, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    return model

def train_lgbm(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv):
    """
    train a gradient boosting LGBM model
    """
    
    evaluation_results = {}
    dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
    dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

    model = lgb.train(lgbm_params,
                      num_boost_round=10000,
                      valid_names=['train', 'valid'],
                      train_set=dtrain,
                      valid_sets=dval,
                      callbacks=[
                          lgb.early_stopping(stopping_rounds=30, verbose=True),
                           lgb.log_evaluation(100),
                          lgb.callback.record_evaluation(evaluation_results)
                        ],
                      )
    return model

def train_linear(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv, model_type='linear'):
    """
    train simple linear model as a baseline
    """

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'lasso':
        model = Lasso(alpha=0.015621759738365798)
    elif model_type == 'ridge':
        model = Ridge(alpha=0.0066619959574408475, random_state=42)
    model.fit(X_train_cv, y_train_cv)    
    return model        

def train_xgb(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv):
    """
    train a gradient boosting XGBoost model
    """
    
    dtrain = xgb.DMatrix(X_train_cv, y_train_cv)
    dvalid = xgb.DMatrix(X_eval_cv, y_eval_cv)
    model = xgb.train(
            xgb_params,
            dtrain,
            10000,
            [(dtrain, "train"), (dvalid, "valid")],
            verbose_eval = 500,
            early_stopping_rounds = 50,
        )
    return model

def train_catboost(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv):
    """
    train a gradient boosting CatBoostRegressor
    """
    
    # Create the CatBoost regressor model
    model = CatBoostRegressor(iterations=296,
                              depth=8,
                              learning_rate=0.09892083096826854,
                              random_strength=6.6234533657114,
                              bagging_temperature=0.09768676525813191,
                              od_type='Iter',
                              od_wait=39,
                              random_state=72,
                              verbose=100)  
    model.fit(X_train_cv, y_train_cv) 
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default=0)   
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model
    
    train = pd.read_csv('25-fold-train.csv')
    columns = [col for col in train.columns if col not in ['kfold', 'monthly_rent']]
    target = 'monthly_rent'    

    models = []
    for fold in range(25):
        print(fold)
        X_train_cv = train[train["kfold"] != fold][columns]
        y_train_cv = train[train["kfold"] != fold][target]        

        X_eval_cv = train[train["kfold"] == fold][columns]
        y_eval_cv = train[train["kfold"] == fold][target]

        X_train_cv = np.array(X_train_cv)
        y_train_cv = np.array(y_train_cv)

        X_eval_cv = np.array(X_eval_cv)
        y_eval_cv = np.array(y_eval_cv)


        if not os.path.exists('models/' + model_name):
            os.mkdir('models/' + model_name)
        
        if 'lgbm' in model_name:
            model = train_lgbm(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv)
        elif 'xgb' in model_name:
            model = train_xgb(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv)
        elif 'tabnet' in model_name:
            model = train_tabnet(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv)
        elif 'catboost' in model_name:
            model = train_catboost(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv)
        elif 'linear' in model_name:
            model = train_linear(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv, 'linear')
        elif 'lasso' in model_name:
            model = train_linear(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv, 'lasso')
        elif 'ridge' in model_name:
            model = train_linear(X_train_cv, y_train_cv, X_eval_cv, y_eval_cv, 'ridge')
    
        model_path = f'models/{model_name}/{fold}.pkl'
        joblib.dump(model, model_path)
        models.append(model)