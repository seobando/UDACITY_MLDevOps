'''
A module to hold the churn functions and libraries

Author: Sebastian Obando Morales
Date: May 15, 2022
'''


# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
#import os
#os.environ['QT_QPA_PLATFORM']='offscreen'

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import constants

def import_data(path_name):
    '''
    returns dataframe for the csv found at path_name

    input:
            path_name: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(path_name)

    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return data_frame


def perform_eda(data_frame, cat_columns, quant_columns):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe
            cat_columns: list of categorical variables names
            quant_columns: list of numerical variables names

    output:
            None
    '''
    # Get categorical distributions
    for category in cat_columns:
        category_name = category.lower()
        image_name_directory = "./images/eda/" + category_name + "_distribution.png"
        plt.figure(figsize=(20, 10))
        data_frame[category].value_counts('normalize').plot(kind='bar')
        plt.savefig(image_name_directory)
        plt.close()
    # Get Quantitive distributions
    for quantity in quant_columns:
        quant_name = quantity.lower()
        if quant_name == "total_trans_ct":
            quant_name = "total_transactions"
        image_name_directory = "./images/eda/" + quant_name + "_distribution.png"
        plt.figure(figsize=(20, 10))
        sns.histplot(data_frame[quantity], stat='density', kde=True)
        plt.savefig(image_name_directory)
        plt.close()
    # Get Head Map
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("./images/eda/heatmap.png")
    plt.close()


def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        feature_lst = []
        if category != 'Churn':
            feature_groups = data_frame.groupby(category).mean()['Churn']
            for val in data_frame[category]:
                feature_lst.append(feature_groups.loc[val])

            data_frame[category + '_Churn'] = feature_lst

    return data_frame


def perform_feature_engineering(data_frame, keep_cols):
    '''
    input:
              data_frame: pandas dataframe
              keep_cols: required columns

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              x_data: pandas dataframe of X selected values for feature importance calculation
    '''
    y_data = data_frame['Churn']
    x_data = pd.DataFrame()
    x_data[keep_cols] = data_frame[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test):
    '''
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data

    output:
              y_train_preds_lr: training predictions from logistic regression
              y_train_preds_rf: training predictions from random forest
              y_test_preds_lr: test predictions from logistic regression
              y_test_preds_rf: test predictions from random forest
    '''
    # Initliaze Random Classifier
    rfc = RandomForestClassifier(random_state=42)
    # Set search grid parameters for random forest
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    # Get best hyperparameters
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # Train the model
    cv_rfc.fit(x_train, y_train)
    # Get train prediction
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    # Get test prediction
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    # Save the model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    # Initliaze Logistic Regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # Train the model
    lrc.fit(x_train, y_train)
    # Get train prediction
    y_train_preds_lr = lrc.predict(x_train)
    # Get test prediction
    y_test_preds_lr = lrc.predict(x_test)
    # Save the model
    joblib.dump(lrc, './models/logistic_model.pkl')

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def classification_report_image(y_train,
                                y_test,
                                x_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            x_test: X testing data
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    models_data = {
        'Random Forest': [
            "./images/results/rf_results.png",
            y_test_preds_rf,
            y_train_preds_rf],
        'Logistic Regression': [
            "./images/results/rf_results.png",
            y_test_preds_lr,
            y_train_preds_lr]}

    for model in models_data:
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, models_data[model][1])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, models_data[model][2])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(models_data[model][0])
        plt.close()

    # Load the models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # ROC curves
    image_name_directory = "./images/results/roc_curve_result.png"
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(rfc_model, x_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(lr_model, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(image_name_directory)
    plt.close()


def feature_importance_plot(model, x_data, output_path_name):
    '''
    creates and stores the feature importances in path_name
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X selected values for feature importance calculation
            output_path_name: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_path_name)
    plt.close()


if __name__ == "__main__":

    # Load constant variables
    PATH_NAME = constants.path_name
    CAT_COLS = constants.cat_columns
    QUANT_COLS = constants.quant_columns
    KEEP_COLS = constants.keep_cols
    print("Phase 0: Import Data\n")
    print("\n-------------------------------------------------------")
    DATA_FRAME = import_data(PATH_NAME)
    print("Phase 1: Perform EDA\n")
    print("\n-------------------------------------------------------")
    perform_eda(DATA_FRAME, CAT_COLS, QUANT_COLS)
    print("Phase 2: Data Preprocessing\n")
    print("\n-------------------------------------------------------")
    DATA_FRAME = encoder_helper(DATA_FRAME, CAT_COLS)
    print("Phase 3: Feature Engineering\n")
    print("\n-------------------------------------------------------")
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DATA_FRAME, KEEP_COLS)
    print("Phase 4: Training\n")
    print("\n-------------------------------------------------------")
    Y_TRAIN_PREDS_LR, Y_TRAIN_PREDS_RF, Y_TEST_PREDS_LR, Y_TEST_PREDS_RF = train_models(
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
    print("Phase 5: Classification reports\n")
    print("\n-------------------------------------------------------")
    classification_report_image(
        Y_TRAIN,
        Y_TEST,
        X_TEST,
        Y_TRAIN_PREDS_LR,
        Y_TRAIN_PREDS_RF,
        Y_TEST_PREDS_LR,
        Y_TEST_PREDS_RF)
    print("Phase 6: Feature Importance\n")
    print("\n-------------------------------------------------------")
    MODEL = joblib.load('./models/rfc_model.pkl')
    X_DATA = pd.DataFrame()
    X_DATA[KEEP_COLS] = DATA_FRAME[KEEP_COLS]
    IMAGE_NAME_DIRECTORY = "images/results/features_importance_plot.png"
    feature_importance_plot(MODEL, X_DATA, IMAGE_NAME_DIRECTORY)
