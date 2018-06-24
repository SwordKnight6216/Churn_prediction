#Author: Gordon Chen
#E-mail: Gordon.Chen@oracle.com
#date: 2018/6/12

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, fbeta_score 
from sklearn.metrics import brier_score_loss

from datetime import datetime


class model_training_eval:
    def __init__(self, classifiers, X_train, X_test, y_train, y_test):
        self.model = []
        self.cls_name = []
        self.cls_roc = []
        self.cls_precision = []
        self.cls_recall = []
        self.cls_f1 = []
        self.cls_brier = []
        self.cls_exec_time = []
        self.cls_hpt_time = []

        self.pred_class = []
        self.pred_proba = []

        self.classifiers = classifiers
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.training_flag = 0

    def _grid_search(self, cls, hpt):
        model_search = GridSearchCV(
            cls, param_grid=hpt, cv=3, verbose=0, n_jobs=1)
        model_search.fit(self.X_train, self.y_train)
        return model_search.best_estimator_

    def training_prediction(self):
        self.training_flag = 1
        
        for cls, hpt in self.classifiers:

            try:
                start_time = datetime.now()
                if hpt:
                    cls = self._grid_search(cls, hpt)
                end_time = datetime.now()
                hpt_time = end_time - start_time

                #training
                start_time = datetime.now()

                cls.fit(self.X_train, self.y_train)

                #prediction
                y_pred_class = cls.predict(self.X_test)
                y_pred_proba = cls.predict_proba(self.X_test)[:, 1]

                end_time = datetime.now()
                exec_time = end_time - start_time

                #Store data for each model
                self.model.append(cls)
                self.cls_name.append(cls.__class__.__name__)

                self._evaluation(cls, y_pred_proba, y_pred_class, exec_time,
                                 hpt_time)

                self.pred_class.append(y_pred_class)
                self.pred_proba.append(y_pred_proba)

                print(
                    "{} is finished after {} HPT. \nThe execution time of the best model is {}.\n".
                    format(cls.__class__.__name__, hpt_time, exec_time))
            except:
                
                print('{} was failed in training'.format(cls.__class__.__name__))

        

    def _evaluation(self, cls, y_pred_proba, y_pred_class, exec_time,
                    hpt_time):

        self.cls_roc.append(roc_auc_score(self.y_test, y_pred_proba))
        self.cls_precision.append(precision_score(self.y_test, y_pred_class))
        self.cls_recall.append(recall_score(self.y_test, y_pred_class))
        self.cls_f1.append(fbeta_score(self.y_test, y_pred_class, 2.0))
        self.cls_brier.append(brier_score_loss(self.y_test, y_pred_proba))
        self.cls_exec_time.append(exec_time)
        self.cls_hpt_time.append(hpt_time)

    def get_performance(self):

        if not self.training_flag:
            self.training_prediction()

        measurement = [
            'roc_auc_score', 'precision_score', 'recall_score', 'fbeta_score',
            'brier_score_loss','execusion_time', 'hpt_time'
        ]
        df_comparason = pd.DataFrame(
            data=[
                self.cls_roc, self.cls_precision, self.cls_recall, self.cls_f1,
                self.cls_brier, self.cls_exec_time, self.cls_hpt_time
            ],
            index=measurement,
            columns=self.cls_name).T

        return df_comparason

    def get_models(self):

        if not self.training_flag:
            self.training_prediction()

        return self.model

    def get_predictions(self):

        if not self.training_flag:
            self.training_prediction()

        return self.cls_name, self.pred_class, self.pred_proba