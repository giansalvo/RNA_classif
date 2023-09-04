#https://www.kaggle.com/code/prashant111/svm-classifier-tutorial
#https://www.kaggle.com/code/rishidamarla/random-forest-classification/notebook
#https://www.kaggle.com/datasets/vinod00725/svm-classification?select=SVM_Classification.R

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # import SVC classifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from matplotlib.colors import ListedColormap
import seaborn as sn
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import datetime

DATASET_FNAME = './PPMI_RNA_IR2.csv'
#DATASET_FNAME = './test.csv'

def main():
    print(f'Start...')
    # Importing the dataset
    #types = defaultdict(float, PATNO="str", subgroup="int")
    #dataset = pd.read_csv(DATASET_FNAME, dtype=types, keep_default_na=False)

    col_names = pd.read_csv(DATASET_FNAME, nrows=0).columns
    types_dict = {'PATNO': str, ' subgroup': int}
    types_dict.update({col: float for col in col_names if col not in types_dict})
    dataset = pd.read_csv(DATASET_FNAME, dtype=types_dict)

    # remove leading spaces from column names
    dataset.columns = dataset.columns.str.strip()
    # drop PATNO column
    dataset = dataset.drop('PATNO', axis=1)

    # Displaying the dataset
    print(dataset.columns)
    print(dataset.head())

    # view the percentage distribution of target_class column
    print("Percentage distribution of target column (subgroup)")
    distrib = dataset['subgroup'].value_counts() / float(len(dataset))
    print(distrib)
    # check null accuracy score
    null_accuracy = (max(distrib[0], distrib[1]) / (distrib[0] + distrib[1]))
    print('Null accuracy score: {0:0.4f}'.format(null_accuracy))

    # check for missing values in variables
    print(dataset.isnull().sum())

    # Splitting the dataset features into X and y
    FEATURES_START = 1   # features start from column 1 (PATNO was removed)
    SUBGROUP_POS = 0    # target is at column 0 (PATNO was removed)
    X = dataset.iloc[0:, FEATURES_START:].values
    y = dataset.iloc[:, SUBGROUP_POS].values
    print(X.shape)
    print(y.shape)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
    print("X_train: {}".format(X_train.shape))
    print("X_val: {}".format(X_val.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_train: {}".format(y_train.shape))
    print("y_val: {}".format(y_val.shape))
    print("y_test: {}".format(y_test.shape))

    # Feature Scaling
    # cols = X_train.columns
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # X_train = pd.DataFrame(X_train, columns=[cols])
    # X_test = pd.DataFrame(X_test, columns=[cols])

    # instantiate classifier with default hyperparameters
    classifier = SVC(kernel='rbf', C=100.0)

    # fit classifier to training set
    classifier.fit(X_train, y_train)

    # print the scores on training and test set
    print('Training set score: {:.4f}'.format(classifier.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(classifier.score(X_test, y_test)))

    # Predicting the Train set results
    print("Performance on train set")
    y_pred = classifier.predict(X_train)
    print("y_pred:{}".format(y_pred.shape))
    # Making the Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)
    print(cm)
    # sn.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    # plt.show()
    # plt.close()
    acc = accuracy_score(y_train, y_pred)
    print("accuracy: {:.4f}".format(acc))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity: {:.4f}'.format(sensitivity))
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('Specificity: {:.4f}'.format(specificity))
    print(classification_report(y_train, y_pred))


    # Predicting the Test set results
    print("Performance on test set")
    y_pred = classifier.predict(X_test)
    print("y_pred:{}".format(y_pred.shape))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # sn.heatmap(cm, annot=True)
    # plt.show()
    # plt.close()
    acc = accuracy_score(y_test, y_pred)
    print("accuracy: {:.4f}".format(acc))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity: {:.4f}'.format(sensitivity))
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('Specificity: {:.4f}'.format(specificity))
    print(classification_report(y_test, y_pred))

    # compute ROC AUC
    ROC_AUC = roc_auc_score(y_test, y_pred)
    print('ROC AUC : {:.4f}'.format(ROC_AUC))

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for Predicting Sporadic Parkinson\'s')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.text(0.7, 0.1, "AUC = {:.4f}".format(ROC_AUC))
    plt.show()

    # compute Stratified k - Fold Cross Validation with shuffle split
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y, cv=kfold)
    print('Stratified cross-validation scores:\n\n{}'.format(scores))
    # print average cross-validation score with rbf kernel
    print('Average stratified cross-validation score with rbf kernel:{:.4f}'.format(scores.mean()))


    # # Visualising the Training set results
    # X_set, y_set = sc.inverse_transform(X_train), y_train
    # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
    #                      np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
    # print(X1.shape)
    # print(X2.shape)
    # print(X1.ravel().shape)
    # print(X2.ravel().shape)
    # sc_transform = sc.transform(np.array([X1.ravel(), X2.ravel()]).T)
    # plt.contourf(X1, X2, classifier.predict(sc_transform).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    # plt.title('Random Forest Classification (Training set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()
    #
    # # Visualising the Test set results
    # X_set, y_set = sc.inverse_transform(X_test), y_test
    # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
    #                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
    # plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    #              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    # plt.title('Random Forest Classification (Test set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()

    begin = datetime.datetime.now().replace(microsecond=0)
    # declare parameters for hyperparameter tuning
    parameters = [{'C': [1, 10, 100, 1000], 'kernel':['linear']},
                    {'C': [1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                    {'C': [1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2, 3, 4], 'gamma':[0.01,0.02,0.03,0.04,0.05]}
                  ]

    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=5,
                               verbose=0)

    grid_search.fit(X_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    grid_search_time = end - begin
    print("GridSearch time: {}\n".format(grid_search_time))

    # examine the best model
    # best score achieved during the GridSearchCV
    print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

    # print parameters that give the best results
    print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

    # print estimator that was chosen by the GridSearch
    print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

    # calculate GridSearch CV score on test set
    print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))


if __name__ == '__main__':
    main()

