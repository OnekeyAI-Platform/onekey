#!/usr/bin/env python
# coding: utf-8

# 

# Model building and evaluation
#data verification and review 
#data regularization which changes the data to obey N~(0, 1)
#construct the training and test sets
#features screening through Lasso and select non-zero items for the subsequent model  
#Use machine learning algorithms for clinical task 
#Model visualization

# feature_file: the path of the feature data
# label_file: label information file for each sample 
# labels: targets to be learned by the AI system

import os
from IPython.display import display
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from onekey_algo import OnekeyDS as okds
import pandas as pd

os.makedirs('img', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('features', exist_ok=True)
# file settings
label_file = r'label information file for each sample'
feature_file = r'the path of your feature data'
labels = ['label']


# 

# read the column name of labeled file 
# read data, the data is stored in CSV forma 
# the required label_data is a 'DataFrame' format including the ID column and the subsequent labels column which can be multiple columns

feature_data = pd.read_csv(feature_file)
display(feature_data)
label_data = pd.read_csv(label_file)
label_data.head()


# feature splicing 
from onekey_algo.custom.utils import print_join_info
print_join_info(feature_data, label_data)
combined_data = pd.merge(feature_data, label_data, on=['ID'], how='inner')
ids = combined_data['ID']
combined_data = combined_data.drop(['ID'], axis=1)
print(combined_data[labels].value_counts())
combined_data.columns


combined_data.describe()


# normalize_df,change the data to a mean of 0,a variance of 1
# 
# $column = \frac{column - mean}{std}$


# regularization
from onekey_algo.custom.components.comp1 import normalize_df
data = normalize_df(combined_data, not_norm=labels, group='group')
data = data.dropna(axis=1)
data.describe()


# correlation coefficient,there are three methods to choose from for calculating the correlation coefficient
   1. pearson: standard correlation coefficient
   2. kendall: Kendall Tau correlation coefficient
   3. spearman: Spearman rank correlation
pearson_corr = data[data['group'] == 'train'][[c for c in data.columns if c not in labels]].corr('pearson')
# kendall_corr = data[[c for c in data.columns if c not in labels]].corr('kendall')
# spearman_corr = data[[c for c in data.columns if c not in labels]].corr('spearman')


# visualization of correlation coefficient
import seaborn as sns
import matplotlib.pyplot as plt
from onekey_algo.custom.components.comp1 import draw_matrix

if combined_data.shape[1] < 100:
    plt.figure(figsize=(50.0, 40.0))
    # select the correlation coefficient for visualization
    draw_matrix(pearson_corr, annot=True, cmap='YlGnBu', cbar=False)
    plt.savefig(f'img/feature_corr.svg', bbox_inches = 'tight')


# cluster analysis 
import seaborn as sns
import matplotlib.pyplot as plt

if combined_data.shape[1] < 100:
    pp = sns.clustermap(pearson_corr, linewidths=.5, figsize=(50.0, 40.0), cmap='YlGnBu')
    plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(f'img/feature_cluster.svg', bbox_inches = 'tight')


# feature filtering--correlation coefficient   
def select_feature(corr threshold: float = 0.9 keep: int = 1 topn=10 verbose=False):
from onekey_algo.custom.components.comp1 
import select_feature
sel_feature = select_feature(pearson_corr, threshold=0.9, topn=32, verbose=False)

sel_feature = sel_feature + labels + ['group']
sel_feature


# features screening
sel_data = data[sel_feature]
sel_data.describe()


# construct datasets

import numpy as np
import onekey_algo.custom.components as okcomp

n_classes = 2
train_data = sel_data[(sel_data['group'] == 'train')]
train_ids = ids[train_data.index]
train_data = train_data.reset_index()
train_data = train_data.drop('index', axis=1)
y_data = train_data[labels]
X_data = train_data.drop(labels + ['group'], axis=1)

test_data = sel_data[sel_data['group'] != 'train']
test_ids = ids[test_data.index]
test_data = test_data.reset_index()
test_data = test_data.drop('index', axis=1)
y_test_data = test_data[labels]
X_test_data = test_data.drop(labels + ['group'], axis=1)

y_all_data = sel_data[labels]
X_all_data = sel_data.drop(labels + ['group'], axis=1)

column_names = X_data.columns
print(f"sample size in the training set：{X_data.shape}, sample size in the validation set：{X_test_data.shape}")


# Lasso, initialize the Lasso model with alpha as the penalty coefficient 
# reference (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?highlight=lasso#sklearn.linear_model.Lasso)
alpha = okcomp.comp1.lasso_cv_coefs(X_data, y_data, column_names=None, alpha_logmin=-3)
plt.savefig(f'img/feature_lasso.svg', bbox_inches = 'tight')


okcomp.comp1.lasso_cv_efficiency(X_data, y_data, points=50, alpha_logmin=-3)
plt.savefig(f'img/feature_mse_label.svg', bbox_inches = 'tight')


# penalty factor, use the penalty factor of cross-validation as the basis for model training
from sklearn import linear_model

models = []
for label in labels:
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_data, y_data[label])
    models.append(clf)


# feature screening, screened features with coef > 0 and print
COEF_THRESHOLD = 1e-8 # feature thresholds after filtering
scores = []
selected_features = []
for label, model in zip(labels, models):
    feat_coef = [(feat_name, coef) for feat_name, coef in zip(column_names, model.coef_) 
                 if COEF_THRESHOLD is None or abs(coef) > COEF_THRESHOLD]
    selected_features.append([feat for feat, _ in feat_coef])
    formula = ' '.join([f"{coef:+.6f} * {feat_name}" for feat_name, coef in feat_coef])
    score = f"{label} = {model.intercept_} {'+' if formula[0] != '-' else ''} {formula}"
    scores.append(score)
    
print(scores[0])


# feature weights
feat_coef = sorted(feat_coef, key=lambda x: x[1])
feat_coef_df = pd.DataFrame(feat_coef, columns=['feature_name', 'Coefficients'])
feat_coef_df.plot(x='feature_name', y='Coefficients', kind='barh')

plt.savefig(f'img/feature_weights.svg', bbox_inches = 'tight')


# screening features, use the features with high coefficients screened by Lasso as training data
X_data = X_data[selected_features[0]]
X_test_data = X_test_data[selected_features[0]]
X_data.columns


# model selection
model_names = ['ExtraTrees','RandomForest','LightGBM','AdaBoost', 'LR', 'MLP']
models = okcomp.comp1.create_clf_model(model_names)
model_names = list(models.keys())


# cross validation (if the study needed, no need in multicenterc ohort)
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import accuracy_score, roc_auc_score

# results = okcomp.comp1.get_bst_split(X_data, y_data, models,  test_size=0.2, metric_fn=roc_auc_score, cv=True, random_state=0)
_, (X_train_sel, X_test_sel, y_train_sel, y_test_sel) = results['results'][results['max_idx']]
# X_train_sel, X_test_sel, y_train_sel, y_test_sel = X_data, X_test_data, y_data, y_test_data
# trails, _ = zip(*results['results'])
# cv_results = pd.DataFrame(trails, columns=model_names)
# sns.boxplot(data=cv_results)
# plt.ylabel('AUC %')
# plt.xlabel('Model Nmae')
# plt.savefig(f'img/model_cv.svg', bbox_inches = 'tight')


# model selection and evaluation
import joblib
from onekey_algo.custom.components.comp1 import plot_feature_importance, plot_learning_curve, smote_resample
targets = []
os.makedirs('models', exist_ok=True)
for l in labels:
    new_models = list(okcomp.comp1.create_clf_model(model_names).values())
    for mn, m in zip(model_names, new_models):
        X_train_smote, y_train_smote = X_train_sel, y_train_sel
#         X_train_smote, y_train_smote = smote_resample(X_train_sel, y_train_sel)
        m.fit(X_train_smote, y_train_smote[l])
        # save result
        joblib.dump(m, f'models/{mn}_{l}.pkl') 
        plot_feature_importance(m, selected_features[0], save_dir='img')
        
#         plot_learning_curve(m, X_train_sel, y_train_sel, title=f'Learning Curve {mn}')
#         plt.savefig(f"img/Rad_{mn}_learning_curve.svg", bbox_inches='tight')
        plt.show()
    targets.append(new_models)


# WSI-level predictions
# predictions for prediction results of each model corresponding to each label.
# pred_scores for predicted probability value for each model for each label.

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from onekey_algo.custom.components.delong import calc_95_CI
from onekey_algo.custom.components.metrics import analysis_pred_binary

predictions = [[(model.predict(X_train_sel), model.predict(X_test_sel)) 
                for model in target] for label, target in zip(labels, targets)]
pred_scores = [[(model.predict_proba(X_train_sel), model.predict_proba(X_test_sel)) 
                for model in target] for label, target in zip(labels, targets)]

metric = []
pred_sel_idx = []
for label, prediction, scores in zip(labels, predictions, pred_scores):
    pred_sel_idx_label = []
    for mname, (train_pred, test_pred), (train_score, test_score) in zip(model_names, prediction, scores):
        acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_train_sel[label], 
                                                                                              train_score[:, 1])
        ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
        metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres, f"{label}-train"))        
      
        acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_test_sel[label], 
                                                                                              test_score[:, 1])
        ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
        metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres, f"{label}-test"))
        pred_sel_idx_label.append(np.logical_or(test_score[:, 0] >= thres, test_score[:, 1] >= thres))
    
    pred_sel_idx.append(pred_sel_idx_label)
metric = pd.DataFrame(metric, index=None, columns=['model_name', 'Accuracy', 'AUC', '95% CI',
                                                   'Sensitivity', 'Specificity', 
                                                   'PPV', 'NPV', 'Precision', 'Recall', 'F1',
                                                   'Threshold', 'Task'])
metric


# draw curves
import seaborn as sns

plt.figure(figsize=(10, 10))
plt.subplot(211)
sns.barplot(x='model_name', y='Accuracy', data=metric, hue='Task')
plt.subplot(212)
sns.lineplot(x='model_name', y='Accuracy', data=metric, hue='Task')
plt.savefig(f'img/model_acc.svg', bbox_inches = 'tight')


# ROC 
sel_model = model_names

for sm in sel_model:
    if sm in model_names:
        sel_model_idx = model_names.index(sm)
    
        # Plot all ROC curves
        plt.figure(figsize=(8, 8))
        for pred_score, label in zip(pred_scores, labels):
            okcomp.comp1.draw_roc([np.array(y_train_sel[label]), np.array(y_test_sel[label])], 
                                  pred_score[sel_model_idx], 
                                  labels=['Train', 'Test'], title=f"Model: {sm}")
            plt.savefig(f'img/model_{sm}_roc.svg', bbox_inches = 'tight')


# model result
sel_model = model_names

for pred_score, label in zip(pred_scores, labels):
    pred_test_scores = []
    for sm in sel_model:
        if sm in model_names:
            sel_model_idx = model_names.index(sm)
            pred_test_scores.append(pred_score[sel_model_idx][1])
    okcomp.comp1.draw_roc([np.array(y_test_sel[label])] * len(pred_test_scores), 
                          pred_test_scores, 
                          labels=sel_model, title=f"Model AUC")
    plt.savefig(f'img/model_roc.svg', bbox_inches = 'tight')


# DCA
from onekey_algo.custom.components.comp1 import plot_DCA

for pred_score, label in zip(pred_scores, labels):
    pred_test_scores = []
    for sm in sel_model:
        if sm in model_names:
            sel_model_idx = model_names.index(sm)
            okcomp.comp1.plot_DCA(pred_score[sel_model_idx][1][:,1], np.array(y_test_sel[label]),
                                  title=f'Rad Model {sm} DCA')
            plt.savefig(f'img/model_{sm}_dca.svg', bbox_inches = 'tight')


# confusion matrix 
# set the drawing parameters
sel_model = model_names
c_matrix = {}

for sm in sel_model:
    if sm in model_names:
        sel_model_idx = model_names.index(sm)
        for idx, label in enumerate(labels):
            cm = okcomp.comp1.calc_confusion_matrix(predictions[idx][sel_model_idx][-1], y_test_sel[label],
#                                                     sel_idx = pred_sel_idx[idx][sel_model_idx],
                                                    class_mapping={1:'1', 0:'0'}, num_classes=2)
            c_matrix[label] = cm
            plt.figure(figsize=(5, 4))
            plt.title(f'Rad Model:{sm}')
            okcomp.comp1.draw_matrix(cm, norm=False, annot=True, cmap='Blues', fmt='.3g')
            plt.savefig(f'img/model_{sm}_cm.svg', bbox_inches = 'tight')


# sample prediction histogram
# plot the predicted results and the corresponding true results for each sample
sel_model = model_names
c_matrix = {}

for sm in sel_model:
    if sm in model_names:
        sel_model_idx = model_names.index(sm)
        for idx, label in enumerate(labels):            
            okcomp.comp1.draw_predict_score(pred_scores[idx][sel_model_idx][-1], y_test_sel[label])
            plt.title(f'{sm} sample predict score')
            plt.legend(labels=["label=0","label=1"],loc="lower right") 
            plt.savefig(f'img/model_{sm}_sample_dis.svg', bbox_inches = 'tight')
            plt.show()


# save  model result
import os
import numpy as np
 
 os.makedirs('results', exist_ok=True)
 sel_model = sel_model
 
 for idx, label in enumerate(labels):
     for sm in sel_model:
         if sm in model_names:
            sel_model_idx = model_names.index(sm)
             target = targets[idx][sel_model_idx]
  # sample prediction result
             train_indexes = np.reshape(np.array(train_ids), (-1, 1)).astype(str)
             test_indexes = np.reshape(np.array(test_ids), (-1, 1)).astype(str)
             y_train_pred_scores = target.predict_proba(X_train_sel)
             y_test_pred_scores = target.predict_proba(X_test_sel)
             columns = ['ID'] + [f"{label}-{i}"for i in range(y_test_pred_scores.shape[1])]
  # save results
             result_train = pd.DataFrame(np.concatenate([train_indexes, y_train_pred_scores], axis=1), columns=columns)
             result_train.to_csv(f'results/{sm}_Rad_train.csv', index=False)
             result_test = pd.DataFrame(np.concatenate([test_indexes, y_test_pred_scores], axis=1), columns=columns)
             result_test.to_csv(f'results/{sm}_Rad_test.csv', index=False)



