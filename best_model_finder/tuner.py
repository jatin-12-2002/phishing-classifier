from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from application_exception.exception import PhishingException
from application_logging.logger import App_Logger

class Model_Finder:
    """
    This class shall  be used to find the model with best accuracy and AUC score.
    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.sv_classifier=SVC()
        self.xgb = XGBClassifier(objective='binary:logistic',n_jobs=-1)
        self.clf = RandomForestClassifier()

    def get_best_params_for_svm(self,train_x,train_y):
        """
        Method Name: get_best_params_for_svm
        Description: get the parameters for the SVM Algorithm which give the best accuracy.
                    Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 
                                        'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"kernel": ['rbf', 'sigmoid'],
                          "C": [0.1, 0.5, 1.0],
                          "gamma": [.001, .1, .4, .004, .003]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.sv_classifier, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.kernel = self.grid.best_params_['kernel']
            self.C = self.grid.best_params_['C']
            self.gamma = self.grid.best_params_['gamma']

            #creating a new model with the best parameters
            self.sv_classifier = SVC(kernel=self.kernel,C=self.C,gamma=self.gamma)
            # training the mew model
            self.sv_classifier.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVM best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svm method of the Model_Finder class')

            return self.sv_classifier
        
        except Exception as e:
            self.logger_object.log(self.file_object,
                        'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                        'SVM training  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise PhishingException(e)

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
        Method Name: get_best_params_for_random_forest
        Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130],
                               "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1),
                               "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                    'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise PhishingException(e)
    
    def get_best_params_for_xgboost(self,train_x,train_y):
        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {
                "n_estimators": [100, 130], 
                "criterion": ['gini', 'entropy'],
                "max_depth": range(7, 10, 1),
                "learning_rate": [0.01, 0.1, 0.2]
            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.learning_rate = self.grid.best_params_['learning_rate']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(criterion=self.criterion,max_depth=self.max_depth,
                                     n_estimators= self.n_estimators,learning_rate=self.learning_rate ,n_jobs=-1 )
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                    'XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        
        except Exception as e:
            self.logger_object.log(self.file_object,
                    'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                    'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise PhishingException(e)

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for Support Vector Classifier
            self.svm=self.get_best_params_for_svm(train_x,train_y)
            self.prediction_svm=self.svm.predict(test_x) # prediction using the SVM Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.svm_score = accuracy_score(test_y,self.prediction_svm)
                self.logger_object.log(self.file_object, 'Accuracy for SVM:' + str(self.svm_score))
            else:
                self.logger_object.log(self.file_object, 'AUC for SVM:' + str(self.svm_score))
            
            # create best model for Random Forest
            self.rf=self.get_best_params_for_random_forest(train_x,train_y)
            self.svm_score = roc_auc_score(test_y, self.prediction_svm) # AUC for SVM
            self.prediction_rf=self.rf.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.rf_score = accuracy_score(test_y,self.prediction_rf)
                self.logger_object.log(self.file_object, 'Accuracy for Random Forest:' + str(self.rf_score))
            else:
                self.rf_score = roc_auc_score(test_y, self.prediction_rf) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for Random Forest:' + str(self.rf_score))

            #comparing the two models
            if(self.svm_score <  self.xgboost_score and self.rf_score < self.xgboost_score):
                return 'XGBoost',self.xgboost
            elif(self.svm_score >  self.xgboost_score and self.svm_score > self.rf_score):
                return 'SVM',self.svm
            else:
                return 'Random Forest',self.rf

        except Exception as e:
            self.logger_object.log(self.file_object,
                    'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                    'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise PhishingException(e)