import pandas as pd
import numpy as np
import pickle


df = pd.read_parquet('./data/rec-exam.parquet', engine = 'pyarrow', columns=['content_id', 'user_id'])
df['click_count'] = df.groupby(['content_id', 'user_id'])['user_id'].transform('size')
df = df[df['click_count']>=10]

df = df.drop_duplicates(subset=['content_id', 'user_id'])
contents = list(set(df['content_id'].tolist()))
contents.sort()

users = list(set(df['user_id'].tolist()))
users.sort()

contents_to_idx_dict = {}
idx_to_contents_dict = {}
for idx, c in enumerate(contents):
    contents_to_idx_dict[c] = idx
    idx_to_contents_dict[idx] = c

users_to_idx_dict = {}
idx_to_users_dict = {}
for idx, u in enumerate(users):
    users_to_idx_dict[u] = idx
    idx_to_users_dict[idx] = u
    
contents = df['content_id'].tolist()
contents_to_idx = []
for c in contents:
    contents_to_idx.append(contents_to_idx_dict[c])

users = df['user_id'].tolist()
users_to_idx = []
for u in users:
    users_to_idx.append(users_to_idx_dict[u])
    
df['content_idx'] = contents_to_idx
df['user_idx'] = users_to_idx
df = df.drop(['content_id', 'user_id'], axis=1)

num_contents = set(contents)
num_users = set(users)
matrix = np.zeros((len(num_contents), len(num_users)))

for i in df.values:
    matrix[i[1]][i[2]] = i[0]


with open('./assets/matrix_data.pkl', 'wb') as f:
    pickle.dump(matrix, f)

with open('./assets/idx_to_contents_dict.pkl', 'wb') as f:
    pickle.dump(idx_to_contents_dict, f)
    
with open('./assets/idx_to_users_dict.pkl', 'wb') as f:
    pickle.dump(idx_to_users_dict, f)
    
    
image_df = pd.read_parquet('./data/rec-exam.parquet', engine = 'pyarrow', columns=['content_id', 'image_url'])
image_df = image_df.drop_duplicates(subset=['content_id'])
with open('./assets/image_df.pkl', 'wb') as f:
    pickle.dump(image_df, f)