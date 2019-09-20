#ML Modules and Models
import tensorflow as tf
from sklearn.linear_model import LogisticRegression, ElasticNetCV, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier, AdaBoostClassifier,VotingClassifier

###XgBoost Model ###
import xgboost as xgb

#MODEL SELECTION, #EVALUATION METRICS
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

#IMBALANCED DATA
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

#Tools
import time


def model_tuner(X_train,y_train,X_dev,y_dev,model,grid = None):

    if grid == None:
        clf = model
    else:
        clf = GridSearchCV(model, grid, cv=10, n_jobs = -1)
    
    clf_fit = clf.fit(X_train,y_train)
    
    if grid != None: 
        best_par = clf.best_params_
    
    y_dev_pred = clf.predict(X_dev)
    y_train_pred = clf.predict(X_train)
    p_pred = clf.predict(X_dev)
    cm = confusion_matrix(y_dev,y_dev_pred)
    dev_accuracy = accuracy_score(y_dev,y_dev_pred)
    train_accuracy = accuracy_score(y_train,y_train_pred)
    report = classification_report(y_dev,y_dev_pred)
    
    if grid != None: 
        print ('\nthe optimal parameters are: {}'.format(best_par))
    
    print ('\naccuracy on the dev set is: {}'.format(dev_accuracy))
    print ('\naccuracy on the train set is: {}'.format(train_accuracy))
    print ('\nconfusion matrix:\n\n{}'.format(cm))
    print ('\nclassification report:\n\n{}'.format(report))
    
    if grid != None:
        results_dict = {'best model':clf_fit,'best parameters':best_par, 'predicted dev values':y_dev_pred, 
                        'predicted training values':y_train_pred,'predicted probabilities':p_pred,
                        'confusion matrix':cm,'dev accuracy':dev_accuracy,
                        'training accuracy':train_accuracy,'classification report':report}
    else:
        results_dict = {'best model':clf_fit, 'predicted dev values':y_dev_pred, 'predicted training values':y_train_pred, 
                        'predicted probabilities':p_pred,'confusion matrix':cm,'dev accuracy':dev_accuracy,
                        'training accuracy':train_accuracy,'classification report':report}
    return results_dict

def seq_data(data, t, time_to_grad, features,balance=False):
    train_t = 'TRAIN{}'.format(str(t))
    dev_t = 'DEV{}'.format(str(t))
    response = 'GRAD_IN_{}.0'.format(time_to_grad)

    y_train = data[train_t][response]
    y_dev = data[dev_t][response]
    X_train = data[train_t][features]
    X_dev = data[dev_t][features]
    
    if balance:
        X_train, y_train = SMOTE().fit_sample(X_train, y_train)
        X_dev, y_dev = SMOTE().fit_sample(X_dev, y_dev)
    
    return (X_train,y_train,X_dev,y_dev)

def accuracy_matrix_fn(df, data):
    aMat = df
    aMat = aMat.append(pd.DataFrame(data), sort=False)
    aMat.reset_index(inplace=True,drop=True)
    return aMat

def run_model(model,X_train,y_train,X_dev,y_dev,grid = None,label=None):
    start = time.time()
    results_log = model_tuner(X_train,y_train,X_dev,y_dev,model,grid)
    end = time.time()
    runtime = end - start
    print ('the runtime is {} minutes'.format(runtime/60))
    
    return {'model':label,'dev accuracy':[results_log['dev accuracy']], 'training accuracy':[results_log['training accuracy']]}

def run_model_xgb(X_train,y_train,X_dev,y_dev,label,num_round):
    
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_dev = xgb.DMatrix(X_dev, label=y_dev)
    
    param = {'max_depth': 6, 'eta': 0.6, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(d_dev, 'eval'), (d_train, 'train')]

    bst = xgb.train(param, d_train, num_round, evallist, verbose_eval=True)
    
    y_dev_pred = (bst.predict(d_dev) > 0.5) * 1
    y_train_pred = (bst.predict(d_train) > 0.5) * 1
    p_pred = bst.predict(d_dev)
    cm = confusion_matrix(y_dev,y_dev_pred)
    dev_accuracy = accuracy_score(y_dev,y_dev_pred)
    train_accuracy = accuracy_score(y_train,y_train_pred)
    dev_f1 = f1_score(y_dev,y_dev_pred)
    train_f1 = f1_score(y_train,y_train_pred)
    dev_precision = precision_score(y_dev,y_dev_pred)
    train_precision = precision_score(y_train,y_train_pred)
    dev_recall = recall_score(y_dev,y_dev_pred)
    train_recall = recall_score(y_train,y_train_pred)
    dev_roc = roc_auc_score(y_dev,y_dev_pred)
    train_roc = roc_auc_score(y_train,y_train_pred)
    report = classification_report(y_dev,y_dev_pred)
    
    print ('\naccuracy on the dev set is: {}'.format(dev_accuracy))
    print ('\naccuracy on the train set is: {}'.format(train_accuracy))
    print ('\nconfusion matrix:\n\n {}'.format(cm))
    print ('\nclassification report:\n\n{}'.format(report))
    
    metrics = {'model':label,
               'dev accuracy':[dev_accuracy], 
               'training accuracy':[train_accuracy],
               'dev f1':[dev_f1],
               'training f1':[train_f1],
               'dev precision':[dev_precision],
               'training precision':[train_precision],
               'dev recall':[dev_recall],
               'training recall':[train_recall],
               'dev roc':[dev_roc],
               'training roc':[train_roc]
              } 
    
    predict = {'predicted dev values':[y_dev_pred],
               'predicted training values':[y_train_pred],
               'predicted probabilities':[p_pred],
               'confusion matrix':[cm],
               'classification report':[report]}
    
    return (metrics, predict)
