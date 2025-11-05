import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


jsons = os.listdir("jsons") # list all the jsons
my_dicts = []
chunk_id = 0



for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"creating Embeddings for {json_file}")  
     
    embeddings = create_embedding([c['text'] for c in content['chunks']])   

    for i, chunk in enumerate(content['chunks']):
        
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        if 'title' not in chunk:
            chunk['title']= f"chunk_{chunk_id}"
        # chunk_id += 1
        my_dicts.append(chunk)
        chunk_id += 1
        
        
            

       
# print(my_dicts)    

df = pd.DataFrame.from_records(my_dicts)
# save the dataframe
joblib.dump(df, 'embeddings.joblib')
incoming_query = input("Ask a questions: ")
question_embedding = create_embedding([incoming_query])[0]


# Find similarities of questions_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_results = 3
max_index = similarities.argsort()[::-1][0:top_results]
print(max_index)
new_df = df.iloc[max_index]
# print(new_df[['number', 'title', 'text']])
# pick only columns that exist
cols_to_show = [c for c in ['chunk_id','number','title','text'] if c in new_df.columns]
print(new_df[cols_to_show])

# show all columns in new_df
pd.set_option('display.max_columns', None)
print(new_df)

