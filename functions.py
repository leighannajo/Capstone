import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
import functools

import warnings
warnings.filterwarnings('ignore')

def test_level(df, encounter='encounter_id', patient='patient_nbr'):
    if len(df) > df['encounter_id'].nunique():
        print("Dataset is probably at the line level.")
    elif len(df) == df['encounter_id'].nunique():
        print("Dataset is probably at the encounter level")
    elif len(df) == df['patient_nbr'].nunique():
        print("Dataset could be at the longitudinal level")
    else:
        print('You did not provide the correct information!')

def missing_data(df):
    """ This function takes in a dataframe, calculates
    the total missing values and the percentage of missing
    values per column and returns a new dataframe.
    """
    new = pd.DataFrame()
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().mean()*100).sort_values(ascending = False)
    new = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return new.loc[new['Percent']>0]

def get_null_get_cardinality(df, categorical_columns):
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
                          grouping_key='patient_nbr'):
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

# referenced from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor, batch_size=32):
    """
    This function takes in a Pandas dataframe, predictor column and batch size.
    It converts the dataframe into Tensorflow datasets.
    Returns TF dataset.
    """
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

def show_transformations(feature_column, example_batch):
    """
    This function takes in a Tensorflow feature and sample batch from
    the original Tensorflow training dataset.
    It returns the feature layer transformations 
    from the sample batch.
    """
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))
    return feature_layer(example_batch)
   
# Categorical Feature Columns:    
def write_vocab_list(vocab_list, column, default_value, 
                vocab_dir='./diabetes_vocab/'): 
    """ 
    This function writes and returns the vocab directory 
    path for the vocab building function.
    """
    output_file_path = os.path.join(vocab_dir, str(column) + "_vocab.txt")
    # put default value in first row 
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0) 
    df = pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, 
                                         header=None)
    return output_file_path


def build_vocab_list(df, categorical_cols, default_value='00'):
    """
     This function takes in the Pandas training dataframe and a list of its 
     categorical columns. 
     It returns a list of the vocab file path for the categorical features.
     """
    vocab_files = []
    for cat in categorical_cols:
        v_file = write_vocab_list(df[cat].unique(), cat, default_value)
        vocab_files.append(v_file)
    return vocab_files


def create_tf_cat_feature(categorical_cols,
                              vocab_dir='./diabetes_vocab/'):
    '''
    This function takes in a list of categorical features to be 
    transformed with TF feature column API and the path where the vocabulary 
    text files are located.
    TF reads from the text files and creates the categorical features.
    It returns the list of transformed TF feature columns.
    '''
    tf_cat_list = []
    for cat in categorical_cols:
        vocab_file_path = os.path.join(vocab_dir,  cat + "_vocab.txt")

        tf_cat_feature = tf.feature_column.categorical_column_with_vocabulary_file(key=cat, 
                            vocabulary_file = vocab_file_path, num_oov_buckets=1)
        tf_cat_feature = tf.feature_column.indicator_column(tf_cat_feature)
        tf_cat_list.append(tf_cat_feature)
    return tf_cat_list

# Numerical Feature Columns:
def normalize_numericals(col, mean, std):
    '''
    This function takes in a column and returns the 
    normalized column.
    '''
    return (col - mean)/std

def create_tf_numerical_feats(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    print(f'### {col}: #mean/std: {MEAN}/{STD}, numeric (normalized)')
    normalizer = functools.partial(normalize_numericals, mean=MEAN, std=STD)
    tf_numerical_feat = tf.feature_column.numeric_column(key=col, 
                    default_value = default_value, normalizer_fn=normalizer, 
                                            dtype=tf.float64)
    return tf_numerical_feat

def calculate_train_stats(df, col):
    """
    This function takes in a dataframe and a column.  It returns
    the mean and std deviation of the column.
    """
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std

def create_tf_numerical_feat_columns(numerical_cols, train_dataframe):
    """
    This function takes a list of numerical columns and the training
    dataset.  It usese the mean and std deviation of each column and 
    creates Tensorflow numerical features.
    It returns those features in an array.
    """
    tf_num_list = []
    for num in numerical_cols:
        mean, std = calculate_train_stats(train_dataframe, num)
        tf_num_feat = create_tf_numerical_feats(num, mean, std)
        tf_num_list.append(tf_num_feat)
    return tf_num_list

'''
Adapted from Tensorflow Probability Regression tutorial  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb    
'''
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2*n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def build_sequential_model(feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dense(75, activation='relu'),
        tfp.layers.DenseVariational(1+1, posterior_mean_field, 
                                    prior_trainable),
        tfp.layers.DistributionLambda(
            lambda t:tfp.distributions.Normal(loc=t[..., :1],
                    scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),])
    return model

def build_model(train_dataset, val_dataset, feature_layer, epochs=5, 
                                                         loss_metric='mse'):
    model = build_sequential_model(feature_layer)
    model.compile(optimizer='rmsprop', loss=loss_metric, metrics=[loss_metric])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=loss_metric, patience=3)     
    history = model.fit(train_dataset, validation_data=val_dataset,
                        callbacks=[early_stop],
                        epochs=epochs)
    return model, history 

def get_mean_std_from_preds(diabetes_yhat):
    '''
    This function takes in a TF Probability prediction object and returns 
    the mean and std dev of the predictions.
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

def get_binary_prediction(df, col):
    '''
    This function takes in a predication dataframe and a 
    probability mean prediction column.  It returns
    flattened numpy array of binary labels.
    '''
    binary_prediction = df[col].apply(lambda x: 1 if x >= 5 else 0).values
    print(f'# Transformed to numpy: {type(binary_prediction)}')
    print(f'Shape: {binary_prediction.shape}')
    return binary_prediction

def add_predictions(df_test, pred_np, demo_col_list, TARGET):
    for cat in demo_col_list:
        df_test[cat] = df_test[cat].astype(str)
    df_test['score'] = pred_np
    df_test['label_value'] = df_test[TARGET].apply(lambda x: 1 if x >=5 else 0)
    return df_test

