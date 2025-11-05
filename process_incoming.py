import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os



def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt, model="llama3.2"):
    r = requests.post("http://localhost:11434/api/generate", json={
        
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return(response)

df = joblib.load('embeddings.joblib')



incoming_query = input("Ask a questions: ")
question_embedding = create_embedding([incoming_query])[0]


# Find similarities of questions_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 3
max_index = similarities.argsort()[::-1][0:top_results]
print(max_index)
new_df = df.iloc[max_index]
# print(new_df[['number', 'title', 'text']])
# pick only columns that exist
# cols_to_show = [c for c in ['chunk_id','number','title','text'] if c in new_df.columns]
# print(new_df[cols_to_show])

# show all columns in new_df
pd.set_option('display.max_columns', None)
print(new_df)

prompt = f'''I am teaching generative ai in generative  course.Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
================================

"{incoming_query}"
user asked this question related to  the video chunks, you have to answer to human way(dont mention the abovee format, its just for you) where and how  much content is taught in which video (in which video  and at what timestamp) and guide the user to go to  that particular video.if user asks unrelated question, tell him that you can only answer questions related to the course
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt, "llama3.2")["response"]
print(response)    

with open("response.txt", "w") as f:
    f.write(response)

print(inference(prompt))    
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
