import pandas as pd
import torch
import torch.nn as nn

# 生成embedding
attributes_df = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTMerged/combined_attributes.csv")
sentence_df = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTChain/combined.csv")

unique_jobs = attributes_df['occupation'].unique()
job_to_id = {job: idx for idx, job in enumerate(unique_jobs)}
attributes_df['occupation'] = attributes_df['occupation'].map(job_to_id)
id_to_attributes = {row['Person id']: row[1:].to_dict() for _, row in attributes_df.iterrows()}

num_occupations = len(unique_jobs)
num_genders = 2
occupation_embed_dim = 64
gender_embed_dim = 8

occupation_embedding = nn.Embedding(num_occupations, occupation_embed_dim)
gender_embedding = nn.Embedding(num_genders, gender_embed_dim)

person_id = sentence_df.iloc[0]['Person id']
attributes = attributes_df[attributes_df['Person id'] == person_id].iloc[0]

occupation_embed = occupation_embedding(torch.tensor([attributes['occupation']]))
gender_embed = gender_embedding(torch.tensor([attributes['gender']]))


