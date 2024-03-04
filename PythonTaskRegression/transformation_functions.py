#!/usr/bin/env python
# coding: utf-8

# In[1]:


def import_data():
    
    """returns the dataset"""
    
    from ucimlrepo import fetch_ucirepo 
    # fetch dataset 
    bike_sharing_dataset = fetch_ucirepo(id=275) 
    X = bike_sharing_dataset.data.features 
    X['cnt'] = bike_sharing_dataset.data.targets
    return X


# In[ ]:


def transform_feature_weathersit(df):
    """
        Args: df (dataframe): the dataframe we will work on
        Returns: df (dataframe) : in the new dataframe value 4 of feature 'weathersit' is replaced by value 3
    """
    df['weathersit']=df['weathersit'].replace(4,3)
    return df


def normalize_lag_values(scaler, df_features, target_values):
    """
        Args: scaler (Standard Scaler)
                df_features   (dataframe): df without normalized lag count values
                target_values  (array): target variable
                
        Return: df_features (dataframe): df_features with normalized lag count values without nan values
                target_values  (array): target variable without nan values
    """
    columns=['lagged_count_24h','lagged_count_25h', 'lagged_count_7d', 'lagged_count_7d1h']
    for column in columns:
        df_features[column]=scaler.transform(df_features[column].values.reshape(-1, 1))
        
    #in order to drop na values firstly we have to join features and target variable together
    df_features['cnt']= target_values
    df_features=df_features.dropna()
    
    target_values=df_features['cnt']
    df_features=df_features.drop('cnt',axis=1)
    
    return df_features,target_values


def create_lagged_values(df):
    
    """
        Args: df (dataframe): the dataframe we will work on
        Returns: df (dataframe): same as input dataframe but with extra lagged values (24h, 25h, 7d, 7d1h) for features   'cnt',  'temp', 'weathersit'
    
    """
    import pandas as pd
    df = pd.concat(
    [
        df,
        df['cnt'].shift(24).rename("lagged_count_24h"),
        df['cnt'].shift(24+1).rename("lagged_count_25h"),
        df['cnt'].shift(7*24).rename("lagged_count_7d"),
        df['cnt'].shift(7*24+1).rename("lagged_count_7d1h"),
        
        df['temp'].shift(24).rename("lagged_temp_24h"),
        df['temp'].shift(24+1).rename("lagged_temp_25h"),
\
        
        df['windspeed'].shift(24).rename("lagged_windspeed_24h"),
        df['windspeed'].shift(24+1).rename("lagged_windspeed_25h"),
        
        df['weathersit'].shift(24).rename("lagged_weathersit_24h"),
        df['weathersit'].shift(24+1).rename("lagged_weathersit_25h"),

    ],
    axis="columns",
    )

    return df




def drop_columns(df_train,df_test=0):
    
    """
        Args: df_train (dataframe): train set
              df_test (dataframe) : test set (optional)
        
        Returns:df_train (dataframe): train set without columns stated in variable 'columns to drop' in the function
              df_test (dataframe) : test set without columns stated in variable 'columns to drop' in the function (in case input id provided)
    
    """
    
    columns_to_drop=['remainder__dteday','remainder__windspeed','remainder__weathersit', 'remainder__temp','remainder__windspeed']
    
    df_train.drop(columns_to_drop,axis=1,inplace=True)
    if df_test != 0:
        df_test.drop(columns_to_drop,axis=1,inplace=True)
        return df_train,df_test
    else:
        return df_train

