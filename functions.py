import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
import functools
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score, f1_score, accuracy_score, roc_curve
from sklearn.metrics import precision_score, make_scorer, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

def test_level(df, encounter='encounter_id', patient='patient_nbr'):
    """[Tests the level of the patient data by checking the 
        unique values of the given columns.  
        Prints a string describing the level.]

    Args:
        df ([dataframe]): [patient data]
        encounter (str, optional): [integer]. Defaults to 'encounter_id'.
        patient (str, optional): [integer]. Defaults to 'patient_nbr'.
    """
    
    if len(df) > df['encounter_id'].nunique():
        print("Dataset is probably at the line level.")
    elif len(df) == df['encounter_id'].nunique():
        print("Dataset is probably at the encounter level")
    elif len(df) == df['patient_nbr'].nunique():
        print("Dataset could be at the longitudinal level")
    else:
        print('You did not provide the correct information!')
        
def plot_PCA_2D(data, target, target_labels, y_colors, n_components=2):
    """[summary]

    Args:
        data ([dataframe]): []
        target ([type]): [feature column]
        target_labels ([type]): [description]
        y_colors ([type]): [description]
        n_components (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(data)
    pcafeatures = pca.transform(data)
    
    for i, label in enumerate(target_labels):
        plt.scatter(pcafeatures[target == i, 0], pcafeatures[target == i, 1],
                  c=y_colors[i], label=label, alpha=.3, edgecolors="none")

    xlabel("1st pricinple component")
    ylabel("2nd pricinple component")
    plt.legend()
    plt.show()
    return pca


def two_plot_PCA_2D(data, target, n_components=2):
    """ This function takes in a dataframe, target feature, 
    feature 
    
    """

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(data)
    pcafeatures = pca.transform(data)
    
    
    plt.scatter(pcafeatures[:,0], pcafeatures[:, 1],
                   alpha=.3, edgecolors="none", c='purple')

    plt.xlabel("1st pricinple component")
    plt.ylabel("2nd pricinple component")
    plt.legend()
    plt.show()
    return pca


def missing_data(df):
    """[Calculates total count of and percentage of null values.]

    Returns:
        [dataframe]: [Columns of null count and percentage null.]
    """
    new = pd.DataFrame()
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().mean()*100).sort_values(ascending = False)
    new = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return new.loc[new['Percent']>0]


"" This function takes in a dataframe and a categorical column list.
    It creates lists for various values and checks for missing, null or 
    placeholder values in each categorical column listed.  It appends the values found
    to this lists and returns dataframes for each """
def get_null_get_cardinality(df, categorical_columns):
    """[summary]

    Args:
        df ([type]): [description]
        categorical_columns ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    col_name = []
    total_unique = []
    placeholders = []
    total_placeholders = []
    col_unique_values = []

    for col in categorical_columns:
        col_name.append(col)
        values = list(df[col].unique())
        col_unique_values.append(values)
        total_unique.append(len(values))

        if_missing = False
        for placeholder in ['?', '?|?', 'None', 'Unknown/Invalid']:
            if placeholder in values:
                num = len(df[(df[col] == placeholder)])
                placeholders.append(placeholder)
                total_placeholders.append(num)
                if_missing = True
                break
        if not if_missing:
            placeholders.append('')
            total_placeholders.append(0)


    df_placeholders = pd.DataFrame({
        'columns': col_name,
        'placeholder': placeholders,
        'total_placeholders': total_placeholders
    })      

    df_cardinality = pd.DataFrame({
        'columns': col_name,
        'total_unique': total_unique,
        'unique_values': col_unique_values
    })  
    return df_placeholders, df_cardinality


def reduce_ndc(df, ndc_df):
    '''
    This function takes in a dataframe and the NDC drug code dataset (with given columns).
    It uses the NDC dataframe to map the generic names with the NDC codes.
    It returns a dataframe with joined generic drug name column.
    '''
    df = pd.merge(df, ndc_df[['NDC_Code', 'Non-proprietary Name']],
                      how="left",
                      left_on='ndc_code',
                      right_on='NDC_Code')
    df.rename(columns={"Non-proprietary Name": "generic_drug"}, inplace=True)
    df.drop(['NDC_Code'], axis=1, inplace=True)
    return df

def select_features(agg_df, cat_col_list, num_col_list, TARGET, 
                          grouping_key='encounter_id'):
    selected_col_list = [grouping_key] + [TARGET] + cat_col_list + num_col_list 
                                
    return agg_df[selected_col_list]

def aggregate_dataset(df, group_list, array_field):
    """
    This function takes in a dataframe, list of columns and a column to 
    aggregate to encounter level.
    It will aggregate given column to a list, create dummy features for 
    variables in the list and then concat the dummy features with the 
    now grouped dataframe.
    It concats while correcting and mapping the column names with zero spaces.
    It returns the new encounter level dataframe and the corresponding
    column name list.
    """
    df = df.groupby(group_list)['encounter_id', 
            array_field].apply(lambda x: x[array_field].values.tolist()).reset_index().rename(columns={
        0: array_field + "_array"}) 
    
    dummy_df = pd.get_dummies(df[array_field + '_array'].apply(pd.Series).stack()).sum(level=0)
    dummy_col_list = [x.replace(" ", "_") for x in list(dummy_df.columns)] 
    mapping_name_dict = dict(zip([x for x in list(dummy_df.columns)], dummy_col_list ) ) 
    concat_df = pd.concat([df, dummy_df], axis=1)
    new_col_list = [x.replace(" ", "_") for x in list(concat_df.columns)] 
    concat_df.columns = new_col_list

    return concat_df, dummy_col_list

def update_dtypes(df, categorical_col_list, predictor):
    df[predictor] = df[predictor].astype(float)
    for col in categorical_col_list:
        df[col] = df[col].astype('str')
    return df
    

def train_test_val_split(df, patient_id):
    '''
    This function takes in a df to be split and the
    patient identifying column.
    It randomly selects the patient data and splits into three
    subsets based on the sample sizes given for each subset.
    It returns:
     - training set: dataframe with 60% of patient data,
     - validation: dataframe with 20% of patient data,
     - test: dataframe with 20% of patient data.
    '''

    df = df.iloc[np.random.permutation(len(df))]
    unique_vals = df[patient_id].unique()
    total_vals = len(unique_vals)
    
    # Split df into train_valid/test (80/20)
    sample_size = round(total_vals * 0.8)
    train_val = df[df[patient_id].isin(unique_vals[:sample_size])]
    train_val.reset_index(drop=True, inplace=True)
    test_df = df[df[patient_id].isin(unique_vals[sample_size:])]
    test_df.reset_index(drop=True)
    
    # Split train into validate/test
    train_size = round(sample_size * 0.75) # 0.8 * 0.75 = 0.6
    train_df = train_val[train_val[patient_id].isin(unique_vals[:train_size])]
    train_df.reset_index(drop=True)
    valid_df = train_val[train_val[patient_id].isin(unique_vals[train_size:])]
    valid_df.reset_index(drop=True)
    return train_df, valid_df, test_df
    
    
def demo_plots(df, predictor):
    """
    This function takes in a dataframe and a target feature and returns
    a plot of the distribution of the target variable across the dataframe.
    """
    print(df.groupby(predictor).size())
    print(df.groupby(predictor).size().plot(kind='barh'))
    
def model_test(model, X_train, y_train, X_test, y_test):
   # fit model on training data
    model.fit(X_train, y_train)
    
    # get model predictions and score on test set
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    return predictions, score

def get_model_performance(predictions, color, X_test, y_test,
                target_names = ['Not Readmitted', 'Readmitted']):
    """[summary]
        
    Args:
        predictions ([type]): [description]
        color ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]
        target_names (list, optional): [description]. Defaults to ['Not Readmitted', 'Readmitted']

    Returns:

    >> run_code()
    """
    conf_matrix_nums = confusion_matrix(y_test, predictions)
    cfm = confusion_matrix(y_test, predictions, normalize='true')
    clr = classification_report(y_test, predictions, target_names=target_names) 
#                     target_names = ['Not Readmitted', '< 30 days',  '> 30 days']
#                                 target_names = ['Not Readmitted', 'Readmitted']
                            
                                           
    true_negative = cfm[0][0]
    false_positive = cfm[0][1]
    false_negative = cfm[1][0]
    true_positive = cfm[1][1]

    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    accuracy = accuracy_score(y_test, predictions)
    
    print(f'Confusion Matrix: \n', conf_matrix_nums, '\n')
    
    print(f'Classification Report: \n', clr, '\n')
    
    print(f'True Negative: {true_negative}')
    print(f'False Positive: {false_positive}')
    print(f'False Negative: {false_negative}')
    print(f'True Positive: {true_positive}', '\n')
    
    print (f'Precision score: {precision}')
    print (f'Recall score: {recall}')
    print (f'F1 score : {f1}')
    print (f'Accuracy score: {accuracy}')
    
    plt.figure(figsize=(13,8))
    sns.heatmap(cfm, 
            annot=True, fmt=".3f", linewidths=.5, square = True, 
                cmap = color);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = (f'Recall Score: {recall}')
    plt.title(all_sample_title, size = 15);
    
def get_important_feats(model, X_train):
    feature_importance = model.feature_importances_
    feat_importances = pd.Series(model.feature_importances_, 
                                 index=X_train.columns)
    feat_importances = feat_importances.nlargest(19)
    feat_importances.plot(kind='barh' , figsize=(10,10)) 
    plt.show()
    

# referenced from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor, **kwargs):
    """
    
    Returns TF dataset.
    """
    y = df[predictor].copy()
    X = df.drop([predictor], axis=1).copy()

    return X, y





