import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def string_conv_int(x):
    mapping = {v: i for i, v in enumerate(set(x))}
    return np.array(list(map(mapping.__getitem__, x)))

def distance_cloest_record(real_df, syn_df):
    # Detect columns with string data
    def encode_integer(df):
        string_columns = df.select_dtypes(include=['object', 'bool']).columns

        # Encode string columns as integers
        for col in string_columns:
            df[col] = string_conv_int(df[col])

        return df

    real_df = real_df.fillna(0)
    syn_df = syn_df.fillna(0)

    real_df = encode_integer(real_df)
    syn_df = encode_integer(syn_df)

    # Convert DataFrames to Dask DataFrames
    real_ddf = dd.from_pandas(real_df, npartitions=5)  # Adjust npartitions based on your available memory
    syn_ddf = dd.from_pandas(syn_df, npartitions=5)    # Adjust npartitions based on your available memory

    # Function to compute the minimum L2 distance for each row in syn_df with respect to real_df
    def compute_min_l2_distance(row, real_array):
        distance_array = np.sqrt(((row.values - real_array)**2).sum(axis=1))
        return np.min(distance_array)

    # Calculate the minimum L2 distance for each row in syn_df with respect to real_df
    real_array = real_ddf.compute().values
    syn_ddf['Min_L2_Distance'] = syn_ddf.map_partitions(lambda part: part.apply(compute_min_l2_distance, axis=1, args=(real_array,)), meta=('Min_L2_Distance', 'f8'))

    # Convert the Dask DataFrame to a Pandas DataFrame
    syn_df_result = syn_ddf.compute()
    min_distances = syn_df_result['Min_L2_Distance']
    
    return min_distances

strings_set = {'abalone', 'adult', 'Bean', 'Churn_Modelling','faults', 'HTRU', 'indian_liver_patient', 
               'insurance', 'Magic', 'nursery', 'Obesity', 'News', 'Shoppers', 'Titanic', 'wilt'}
Model = {'TabAutoDiff'}

DCR_mean = np.zeros((len(Model), len(strings_set)))
DCR_var = np.zeros((len(Model), len(strings_set)))

data_idx = 0
for string in strings_set:
    print(string)
    
    real_file = f'C:/Users/Namjoon Suh/Desktop/Tabular/Real-data/{string}.csv' 
    real_df = pd.read_csv(real_file)
    
    model_idx = 0
    for model in Model:
        print(model)
        
        empty_mean = []

        for i in range(1,11):
            syn_file = f'C:/Users/Namjoon Suh/Desktop/Tabular/Synthetic-data/{model}/{string}/AutoDiff_{string}{i}.csv'
        
            syn_df = pd.read_csv(syn_file)
            Min_L2_Dist = distance_cloest_record(real_df, syn_df)
            empty_mean.append(np.mean(Min_L2_Dist))
            
        DCR_mean[model_idx][data_idx] = np.mean(empty_mean)
        DCR_var[model_idx][data_idx] = np.mean(empty_mean)
        
        model_idx += 1
        
    data_idx += 1

#df.to_csv('file_name.csv')




