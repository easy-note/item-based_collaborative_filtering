import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib.image as img
import os

with open('./assets/matrix_data.pkl', 'rb') as f:
    matrix_data = pickle.load(f)

with open('./assets/idx_to_contents_dict.pkl', 'rb') as f:
    idx_to_contents_dict = pickle.load(f)
    
with open('./assets/image_df.pkl', 'rb') as f:
    image_df = pickle.load(f)
    
def get_contents_id(idx_id):
    return idx_to_contents_dict[idx_id]

def draw_img(target, recommends):
    plt.figure(figsize=(20,20))
     
    target_img_url = image_df[image_df['content_id']==target]['image_url'].iloc[0]
    os.system("curl " + target_img_url + " > ./assets/target_img.jpg")
    
    for idx, i in enumerate(recommends):
        recommend_img_url = image_df[image_df['content_id']==i]['image_url'].iloc[0]
        os.system("curl " + recommend_img_url + " > ./assets/recommend_img_{}.jpg".format(idx))
       
    image = img.imread('./assets/target_img.jpg')
    plt.subplot(3,4,1)
    plt.imshow(image)
    
    sub = [5,6,7,8,9,10,11,12]
    for i in range(8):
        image = img.imread('./assets/recommend_img_{}.jpg'.format(i))
        plt.subplot(3,4,sub[i])
        plt.imshow(image)
    
    plt.savefig('test3.jpg')
    
    
matrix_data = pd.DataFrame(matrix_data)

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(matrix_data.values)
distances, indices = knn.kneighbors(matrix_data.values, n_neighbors=9)

idx_for_content = matrix_data.index.tolist().index(0)
sim_contents = indices[idx_for_content].tolist()
id_movie = sim_contents.index(idx_for_content)
sim_contents.remove(idx_for_content)


target_content = get_contents_id(0)
recommend_contents = []
for i in sim_contents:
    recommend_contents.append(get_contents_id(i))
    
print('The Nearest Content to {} : {}'.format(target_content, recommend_contents))

draw_img(target_content, recommend_contents)

