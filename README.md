# CS5228-Project
All of the models were trained using GPU expect scikit-learn model "Linear Regression and Ridge Regression". The final submission achieved a 25-fold CV score of 481.1757989 and LB score of 478.96801.

## Model overview of our final submission

| Model * (Count) | CV (25-folds) | Notes
| --- | --- | --- | 
| LGBM * 3 | 482.55819 | Hyperparameter tuning and seed averaging |
| CatboostRegressor * 4 | 483.79348 | Hyperparameter tuning and seed averaging |
| Tabnet * 2 | 484.5554 | Hyperparameter tuning and seed averaging |
| XGBoost * 2 | 484.5425 | Hyperparameter tuning and seed averaging |
| Linear Regression | 491.6013 | Hyperparameter tuning |
| Ridge Regression | 491.6 | Hyperparameter tuning |
| <b>Final Ensemble Score</b> | <b>481.1757989</b> | <b>Weighted averaging of all models</b> |

#### Notes
* <a href='https://optuna.org/'>Optuna</a> is used to search for the best parameters for all the models.
* Notes: Ensure the machine has at CUDA capable device with at least 4GB of VRAM

## Project structure 
* `pipeline/train` folder: code for training the different models
    - `models/`: Directory of the train model weights are stored here
    - `train.py`: Training code to train the different models
    - `25-fold-train.csv`: A preprocessed training csv are provided, to allow for training of different model since preprocessing take time
* `pipeline/inference` folder: training and inference code for the GPR model
    - `models/`: The trained model weights are stored here
    - `inference.py`: Perform inference on the test dataframe of 30k records and generate a submission.csv
    - `test_df.csv`: A preprocessed test csv are provided, to speed up the inference time since preprocessing take time
* `pipeline/utils` folder: utiliy and constant files for data preprocessing
    - `preprocessing.py`: Custom preprocessing function on dataframe
    - `const.py`: Constants for storing statistics of the training data and optimal hyperparameters of the models

## Hardware 
* Memory: 64 GB
* GPU: RTX 3090 

## Software
* Ubuntu 22.04 with Linux 5.8.0
* CUDA: 11.7
* Python: 3.8
* Library dependencies: see the `requirements.txt` for details.

## Environment setup
```
$ conda create --name cs5228_project python=3.8
$ conda activate cs5228_project
$ pip install -r requirements.txt
```

## Training
1. Train the following [list of models](#Model-overview-of-our-final-submission).
e.g. To train a specific model run the following command
    ```
    > python train.py --model "lgbm_models"
    > python train.py --model "xgb_models"
    > python train.py --model "tabnet_models"    
    > python train.py --model "catboost_models"
    > python train.py --model "linear_models"
    > python train.py --model "lasso_models"
    > python train.py --model "ridge_models"
    ```
    This will train a specific model type for 25 times (25-folds). After training the model weights will be stored at `train/models/lgbm_models/0..24.pkl`
    `train/models/xgb_models/0..24.pkl`
    etc..


## Inference
* Use `inference.py` in the `inference` folder to run inference on the preprocessed test dataframe `test_df.csv`.
* This will run the [ensemble model](#Model-overview-of-our-final-submission) using the optimial weights to combine the individual model predictions using weighted averaging.
    ```
    > python inference.py
    ```
* After running inference, it will generate a `submission.csv` in the same directory    

### Team 49
| Team member | Workload |
| --- | --- |
| Ding Ming | Preprocessing, Model selection & evaluation |
| Kelvin Soh Boon Kai | Preprocessing, Model training/inference pipeline creation, Model selection & evaluation |
| Pan Jiangdong | EDA, Feature Engineering |
| Royce Lim Jie Han | EDA, Feature Engineering |

