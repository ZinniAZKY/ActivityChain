import pandas as pd
import torch.nn as nn
import torch


# # 重新设置id并合并csv
# def update_ids(main_df, attributes_df, max_id_previous):
#     main_df['Person id'] = main_df['Person id'].astype(int) + max_id_previous
#     attributes_df['Person id'] = attributes_df['Person id'].astype(int) + max_id_previous
#     return main_df, attributes_df
#
#
# def harmonize_jobs(df, mapping):
#     df['occupation'] = df['occupation'].map(mapping)
#     return df
#
#
# mapping_file1 = {
#     1: "governor",
#     2: "technician",
#     3: "officer",
#     4: "seller",
#     5: "servicer",
#     6: "guards",
#     7: "farmer",
#     8: "engineer",
#     9: "deliver",
#     10: "builder",
#     11: "cleaner",
#     12: "other",
#     13: "junior_student",
#     14: "senior_student",
#     15: "house_wife",
#     16: "non_worker",
#     99: "other",
# }
#
# mapping_file2 = {
#     1: "governor",
#     2: "technician",
#     3: "officer",
#     4: "seller",
#     5: "servicer",
#     6: "guards",
#     7: "farmer",
#     8: "engineer",
#     9: "deliver",
#     10: "builder",
#     11: "cleaner",
#     12: "other",
#     21: "other",
#     13: "junior_student",
#     14: "senior_student",
#     15: "house_wife",
#     16: "non_worker",
#     99: "other",
# }
#
# mapping_file3 = {
#     1: "farmer",
#     2: "engineer",
#     3: "seller",
#     4: "servicer",
#     5: "deliver",
#     6: "guards",
#     7: "officer",
#     8: "technician",
#     9: "governor",
#     10: "other",
#     11: "junior_student",
#     12: "senior_student",
#     13: "senior_student",
#     14: "house_wife",
#     15: "non_worker",
#     16: "other",
#     99: "other",
# }
#
# df1 = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTChain/Chukyo2011PTChain.csv")
# df2 = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTChain/Kinki2010PTChain.csv")
# df3 = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTChain/Tokyo2008PTChain.csv")
#
# attr_df1 = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTMerged/Chukyo2011PTMerged.csv")
# attr_df2 = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTMerged/Kinki2010PTMerged.csv")
# attr_df3 = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTMerged/Tokyo2008PTMerged.csv")
#
# attr_df1 = harmonize_jobs(attr_df1, mapping_file1)
# attr_df2 = harmonize_jobs(attr_df2, mapping_file2)
# attr_df3 = harmonize_jobs(attr_df3, mapping_file3)
#
# df2, attr_df2 = update_ids(df2, attr_df2, df1['Person id'].max())
# df3, attr_df3 = update_ids(df3, attr_df3, df2['Person id'].max())
#
# combined_df = pd.concat([df1, df2, df3], ignore_index=True)
# combined_attr_df = pd.concat([attr_df1, attr_df2, attr_df3], ignore_index=True)
#
# combined_df.to_csv("/Users/zhangkunyi/Downloads/PTFolder/PTChain/combined.csv", index=False)
# combined_attr_df.to_csv("/Users/zhangkunyi/Downloads/PTFolder/PTMerged/combined_attributes.csv", index=False)



# # 转换csv为txt
# df = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTChain/combined.csv")
#
# columns_to_retain = [df.columns[0]] + list(df.columns[25:73])
# df = df[columns_to_retain]
#
# for col in df.columns[1:]:
#     df[col] = df[col].replace({46: "Private_Movement", 47: "Go_Other_Business"})
#
# with open("/Users/zhangkunyi/Downloads/PTFolder/PTChain/combined.txt", "w") as f:
#     for index, row in df.iterrows():
#         f.write(" ".join(row.astype(str)) + "\n")



# 生成embedding
attributes_df = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTMerged/combined_attributes.csv")
sentence_df = pd.read_csv("/Users/zhangkunyi/Downloads/PTFolder/PTChain/combined.csv")

unique_jobs = attributes_df['occupation'].unique()
job_to_id = {job: idx for idx, job in enumerate(unique_jobs)}
attributes_df['occupation'] = attributes_df['occupation'].map(job_to_id)
attributes_df = attributes_df.drop_duplicates(subset='Person id').set_index('Person id')
id_to_attributes = attributes_df.to_dict(orient='index')

num_occupations = len(unique_jobs)
num_genders = 2
occupation_embed_dim = 64
gender_embed_dim = 8

occupation_embedding = nn.Embedding(num_occupations, occupation_embed_dim)
gender_embedding = nn.Embedding(num_genders, gender_embed_dim)


# if __name__ == "__main__":
#     # Display unique jobs and their IDs
#     print("Job to ID mapping:\n", job_to_id)
#
#     # Display attributes for a few sample person IDs
#     sample_ids = [1, 2, 3]  # Modify this based on your dataset
#     for person_id in sample_ids:
#         print(f"\nAttributes for Person ID {person_id}:")
#         print(id_to_attributes[person_id])
#
#     # Display embeddings for a few sample occupations and genders
#     sample_occupations = [0, 1]  # Modify this based on your dataset
#     sample_genders = [0, 1]     # Assuming 0 and 1 represent the two genders
#     for occupation in sample_occupations:
#         occupation_embed_sample = occupation_embedding(torch.tensor([occupation]))
#         print(f"\nOccupation Embedding for ID {occupation}:")
#         print(occupation_embed_sample)
#
#     for gender in sample_genders:
#         gender_embed_sample = gender_embedding(torch.tensor([gender]))
#         print(f"\nGender Embedding for ID {gender}:")
#         print(gender_embed_sample)

