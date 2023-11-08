# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:59:16 2023
@author: Namjoon Suh
"""
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

def detect_column_types(dataframe):
    continuous_columns = []
    categorical_columns = []

    for col in dataframe.columns:
        # Calculate the ratio of unique values to the total number of rows
        n_unique = dataframe[col].nunique()

        # If the ratio is below the threshold, consider the column as categorical
        if n_unique >= 2 and n_unique <= 20:
            categorical_columns.append(col)
        elif dataframe[col].dtype == 'object':
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)

    return continuous_columns, categorical_columns

def string_conv_int(x):
    mapping = {v: i for i, v in enumerate(set(x))}
    return np.array(list(map(mapping.__getitem__, x)))

strings_set = {'abalone', 'adult', 'Bean', 'Churn_Modelling','faults', 'HTRU', 'indian_liver_patient', 
               'insurance', 'Magic', 'nursery', 'Obesity', 'News', 'Shoppers', 'Titanic', 'wilt'}
Model = {'AutoDiff', 'TabAutoDiff', 'NewAutoDiff', 'NewAutoDiff', 'AutoGAN', 'CTAB-GAN-Plus', 'CTGAN', 'Stasy', 'TVAE', 'TabDDPM'}

final_cont_rank = np.zeros(len(Model))
final_cate_rank = np.zeros(len(Model))
final_tota_rank = np.zeros(len(Model))

for string in strings_set:
    print(string)
    real_file = f'C:/Users/Namjoon Suh/Desktop/Tabular/Real-data/{string}.csv'
    real_df = pd.read_csv(real_file)
    
    continuous_columns, categorical_columns = detect_column_types(real_df)
    
    cont_rank = np.empty( (10, len(continuous_columns)) )
    cate_rank = np.empty( (10, len(categorical_columns)) )
    tota_rank = np.empty( (10, len(continuous_columns) + len(categorical_columns)) )
    
    model_cont = np.empty( (len(Model), len(continuous_columns)) )
    model_cate = np.empty( (len(Model), len(categorical_columns)) )   
    model_tota = np.empty( (len(Model), len(continuous_columns) + len(categorical_columns)) )   

    mod_idx = 0
    
    for model in Model:
        print(model)
        for i in range(1,11):

            if model == 'TabAutoDiff':
                syn_file = f'C:/Users/Namjoon Suh/Desktop/Tabular/Synthetic-data/{model}/{string}/AutoDiff_{string}{i}.csv'        
            else:
                syn_file = f'C:/Users/Namjoon Suh/Desktop/Tabular/Synthetic-data/{model}/{string}/{model}_{string}{i}.csv'        

            syn_df = pd.read_csv(syn_file)
            
            cont_col_idx = 0
            cate_col_idx = 0

            for col in real_df.columns:
                if col in continuous_columns:
                    real_col = real_df[col].fillna(real_df[col].mean())
                    syn_col = syn_df[col].fillna(syn_df[col].mean())
                    
                    cont_rank[i-1][cont_col_idx] = wasserstein_distance(real_col, syn_col)
                    cont_col_idx += 1

                else:
                    real_col = real_df[col].fillna(real_df[col].mode()[0])
                    syn_col = syn_df[col].fillna(syn_df[col].mode()[0])                
        
                    if syn_df[col].dtype == 'object':
                        real_col = string_conv_int(real_col)
                        syn_col = string_conv_int(syn_col)
        
                    cate_rank[i-1][cate_col_idx] = jensenshannon(real_col, syn_col)
                    cate_col_idx += 1

        model_cont[mod_idx] = np.mean(cont_rank, axis=0)
        model_cate[mod_idx] = np.mean(cate_rank, axis=0)
        model_tota[mod_idx] = np.mean(np.concatenate((cont_rank, cate_rank), axis=1), axis=0)        

        mod_idx += 1

    cont_ranks = np.argsort(np.argsort(model_cont, axis=0), axis=0) + 1
    cont_ranks = np.mean(cont_ranks, axis=1)
        
    cate_ranks = np.argsort(np.argsort(model_cate, axis=0), axis=0) + 1
    cate_ranks = np.mean(cate_ranks, axis=1)
    
    tota_ranks = np.argsort(np.argsort(model_tota, axis=0), axis=0) + 1
    tota_ranks = np.mean(tota_ranks, axis=1)
    
    cont_ranks = np.nan_to_num(cont_ranks, nan=3)
    cate_ranks = np.nan_to_num(cate_ranks, nan=3)
    tota_ranks = np.nan_to_num(tota_ranks, nan=3)
    
    final_cont_rank = final_cont_rank + cont_ranks/len(strings_set)
    final_cate_rank = final_cate_rank + cate_ranks/len(strings_set)
    final_tota_rank = final_tota_rank + tota_ranks/len(strings_set)


