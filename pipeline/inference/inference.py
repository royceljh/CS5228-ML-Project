import sys
import os
import joblib
import xgboost as xgb
sys.path.append('../../')
sys.path.append('../')
from utils.preprocessing import *
from utils.const import *

def preprocess_test_df(df=None):
    """
        preprocess dataframe to generate features
        df: Dataframe
    """
    test = pd.read_csv('../../data/test.csv')
    test = preprocess_geographic_location_pang(test)
    test['rent_approval_year'] = test['rent_approval_date'].apply(lambda x: x.split('-')[0])
    test['rent_approval_month'] = test['rent_approval_date'].apply(lambda x: x.split('-')[1])
    test['year_month_adjusted_close'] = test.apply(custom_func_adjusted_close, axis=1)

    clean_replace_numeric(test, 'flat_type', '2', 2)
    clean_replace_numeric(test, 'flat_type', '3', 3)
    clean_replace_numeric(test, 'flat_type', '4', 4)
    clean_replace_numeric(test, 'flat_type', '5', 5)
    clean_replace_numeric(test, 'flat_type', 'executive', 6)

    test_town_dummies = generate_dummies(test, 'town')
    test_flat_model_dummies = generate_dummies(test, 'flat_model')
    # train['flat_type'] = train['flat_type'].astype(int)
    test_flat_type_dummies = generate_dummies(test, 'flat_type')
    test_lease_commence_dummies = generate_dummies(test, 'lease_commence_date')
    test_region_dummies = generate_dummies(test, 'region')
    test_rent_approval_year_dummies = generate_dummies(test, 'rent_approval_year')
    test_rent_approval_month_dummies = generate_dummies(test, 'rent_approval_month')
    test_subzone_dummies = generate_dummies(test, 'subzone')
    test_planning_area_dummies = generate_dummies(test, 'planning_area')
    # no_mrt_dummies = generate_dummies(df, 'no_mrt')

    test['year_month_monthly_coe_mean'] = test.apply(custom_func, axis=1)
    test['year_month_monthly_quota_mean'] = test.apply(custom_func_quota, axis=1)
    test['year_month_monthly_bids_mean'] = test.apply(custom_func_bids, axis=1)

    for c in set(train_flat_model_dummies) - set(test_flat_model_dummies.columns):
        test_flat_model_dummies[c] = 0

    test['year_month_open'] = test.apply(custom_func_open, axis=1)
    test['year_month_high'] = test.apply(custom_func_high, axis=1)
    test['year_month_low'] = test.apply(custom_func_low, axis=1)
    test['year_month_close'] = test.apply(custom_func_close, axis=1)

    test_df = test.copy()

    test = pd.concat([
                    test_region_dummies, test_town_dummies, test_flat_model_dummies, test_flat_type_dummies, test_lease_commence_dummies,
                    test_rent_approval_year_dummies, test_rent_approval_month_dummies, test_subzone_dummies,
                   test_df[['floor_area_sqm', 'dist_mrt_exist_pang', 'dist_mrt_planned_pang', 'dist_primary_school_pang', 'dist_shopping_malls_pang',
                        'year_month_monthly_coe_mean', 'year_month_monthly_quota_mean', 'year_month_monthly_bids_mean',
                         'year_month_adjusted_close'#, 'year_month_open', 'year_month_high', 'year_month_low', 'year_month_close'
                        ]],
              ], axis=1)
    
    test = test.rename(columns={'dist_mrt_exist_pang': 'dist_mrt_exist',
                    'dist_mrt_planned_pang': 'dist_mrt_planned',
                    'dist_primary_school_pang': 'dist_primary_school',
                    'dist_shopping_malls_pang': 'dist_shopping_malls'})
    
    return test


def run_inference(test, model_name='lgbm_models', n_fold=25):
    '''
        perform inference on test dataframe
        test: Dataframe
        model_name: name of the "models" to use, can be found in directory models/
        n_folds: number of training folds to use for inference, default is 25
    '''
    preds_test = []
    for fold in range(25):
        X_eval_cv = test
        model_path = f'models/{model_name}/{fold}.pkl'
        if 'xgb' in model_path:
            X_eval_cv = xgb.DMatrix(X_eval_cv)
        elif 'tab' in model_path:
            X_eval_cv = np.array(X_eval_cv)
        model = joblib.load(model_path)
        pred_test = model.predict(X_eval_cv).flatten()
        preds_test.append(pred_test)
        
    preds_test = np.mean(preds_test, axis=0)
    return preds_test

if __name__ == '__main__':
    train = pd.read_csv('../train/25-fold-train.csv')
    columns = [col for col in train.columns if col not in ['kfold', 'monthly_rent']]
    # test_df = preprocess_test_df()
    test_df = pd.read_csv('test_df.csv')
    # test_df.to_csv('test_df.csv', index=False)
    
    lgbm_models_path = ['lgbm_models', 'lgbm_models2', 'lgbm_models3']
    xgb_models_path = ['xgb_models', 'xgb_models2']
    ridge_models_path = ['ridge_models']
    tabnet_models_path = ['tabnet_models', 'tabnet_models2']
    cat_models_path = ['cat_models', 'cat_models2', 'cat_models3', 'cat_models4']
    lr_models_path = ['lr_models']
        
    print('running inference..')
    
    # weights for the different model for ensemble weighted averaging
    weights_list = [.436356, .320497, -.00799, .07928, .322106, -.148752]
    preds_final = np.zeros(len(test_df))
    final_preds_list = []
    for weight, model_paths in zip(weights_list, [lgbm_models_path, xgb_models_path, ridge_models_path,
                        lr_models_path, tabnet_models_path, cat_models_path]):
        preds_list = []
        for model_name in model_paths:
            preds = run_inference(test_df[columns], model_name)
            preds_list.append(preds)
        preds_final += np.mean(preds_list, axis=0)*weight
    
    test_df['Predicted'] = preds_final
    submission = test_df[['Predicted']].reset_index().rename(columns={'index': 'Id'})
    submission.to_csv('submission.csv', index=False)
    print(preds_final, preds_final.sum())
    