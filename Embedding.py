import pandas as pd
import torch.nn as nn
import torch

# 生成embeddings
attributes_df = pd.read_csv("/home/ubuntu/Documents/TokyoPT/combined_spatial_personal.csv")

unique_jobs = attributes_df['occupation'].unique()
attributes_df = attributes_df.drop_duplicates(subset='Person id').set_index('Person id')
id_to_attributes = attributes_df.to_dict(orient='index')

num_genders = 2
num_age = 19
num_occupations = len(unique_jobs)
num_environment = 5
num_traffic = 4

gender_embed_dim = 2
age_embed_dim = 4
occupation_embed_dim = 4
environment_embed_dim = 2
traffic_embed_dim = 2
