#https://www.kaggle.com/code/rishidamarla/random-forest-classification/notebook

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
import seaborn as sn
from collections import defaultdict

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
    print(dataset['subgroup'].value_counts() / float(len(dataset)))

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
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the Random Forest Classification model on the Training set
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=20, random_state=0)
    classifier.fit(X_train, y_train)

    predictions = []
    for tree in classifier.estimators_:
        predictions.append(tree.predict_proba(X_val)[None, :])
    predictions = np.vstack(predictions)
    cum_mean = np.cumsum(predictions, axis=0) / np.arange(1, predictions.shape[0] + 1)[:, None, None]
    scores = []
    for pred in cum_mean:
        scores.append(accuracy_score(y_val, np.argmax(pred, axis=1)))
    plt.figure(figsize=(10, 6))
    plt.plot(scores, linewidth=3)
    plt.xlabel('num_trees')
    plt.ylabel('accuracy');
    # plt.show()    # DEBUG UNCOMMENTO TO  SHOW THE GRAPH


    # Predicting the Validation  set results
    print("Performance on validation set")
    y_pred = classifier.predict(X_val)
    print("y_pred:{}".format(y_pred.shape))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    # sn.heatmap(cm, annot=True)
    # plt.show()
    # plt.close()
    acc = accuracy_score(y_val, y_pred)
    print("accuracy: {:.4f}".format(acc))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity: {:.4f}'.format(sensitivity))
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('Specificity: {:.4f}'.format(specificity))


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

if __name__ == '__main__':
    main()

