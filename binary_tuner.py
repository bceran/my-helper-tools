from datetime import datetime, timedelta
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np

class LgbTuning:
    def __init__(self, dataframe, target, n_iter):
        self.study = optuna.create_study(direction="maximize")
        self.best_model = None
        self.dtrain = lgb.Dataset(dataframe.drop([target], axis=1), 
                                  label=dataframe[target])
        
        self.n_iter = n_iter
        self.best_accuracy = 0
        self.best_params = {}

    def objective(self, trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'verbosity':-1,
            'num_leaves': trial.suggest_int('num_leaves', 2, 128),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 2, 10),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        }

        skf = StratifiedKFold(n_splits=8, random_state=773, shuffle=True)
        
        lgbcv = lgb.cv(param,
                       self.dtrain,
                       folds=skf,
                       verbose_eval=False,
                       num_boost_round=self.n_iter,
                       return_cvbooster=True,
                       )

        cv_score = lgbcv['binary_error-mean'][-1]
        iter_accuracy_score = 1-lgbcv['binary_error-mean'][-1]
            self.best_accuracy = iter_accuracy_score
            print(f"Best CV : {self.best_accuracy}")

        return iter_accuracy_score

    def start_tuning(self, n_minutes):
        now = datetime.now()
        end_time = now + timedelta(minutes = n_minutes)
        end_time = end_time.strftime("%H:%M")
        print("End Time =", end_time)
        
        optuna.logging.set_verbosity(optuna.logging.ERROR) 
        self.study.optimize(self.objective, timeout=n_minutes*60)
        best_params = {
                'objective': 'binary',
                'metric': 'binary_error',
                'boosting_type': 'gbdt',
                'verbosity':-1,
        } 
        best_params.update(self.study.best_params)
        self.best_model = lgb.train(best_params,
                                    self.dtrain,
                                    verbose_eval=False,                   
                                    num_boost_round=self.n_iter,
                                   )
        self.best_params = best_params

    def pred(self, test_df):
        if self.best_model == None:
            raise AttributeError("Best_model is not exist. Please finish tuning operation.")
        preds = np.round(self.best_model.predict(test_df)).astype(np.int8)

        return preds