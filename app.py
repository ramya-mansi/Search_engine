from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import scipy
import time

# Create a Flask application
app = Flask(__name__)

# Load the pre-trained model
model = SentenceTransformer('paraphrase-distilroberta-base-v2')

# Load the data.csv file into a Pandas dataframe
df = pd.read_csv('C:/Users/ramya/OneDrive/Desktop/SSE/Search_engine/topics.csv')

cleaned_data_embeddings = model.encode(df['Cleaned Data'].values.tolist(), convert_to_tensor=True)
cleaned_data_embeddings = cleaned_data_embeddings.cpu().float()

# Define a route for the chat interface
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to process user messages
@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    user_message = data['message']
    
    if user_message == 'hi':
        bot_response = 'Hello, How are you?'
    elif 'how are you' in user_message or 'what about you' in user_message:
        bot_response = 'I am good'
    elif user_message == 'How is the weather?':
        bot_response = 'It is Sunny today'
    elif user_message == 'What is the date today?':
        bot_response = str(datetime.now().date())
    else:
        # Handle semantic search here
        print('hi')
        start_time = time.time()
        query_embedding = model.encode(user_message, convert_to_tensor=True).cpu().float()
        
        search_results = semantic_search_results(query_embedding, number_top_matches=3)
        print('done')
        end_time = time.time()
        # Calculate the elapsed time
        bot_response = Markup("Here are some related documents:<br>")
        for result in search_results:
            bot_response += Markup(f"- {result['Title']} {result['Reference']} ({result['Information']})<br>")
        print(end_time - start_time)
    
    return jsonify({'response': bot_response})

def semantic_search_results(query_embedding, query_df=None, number_top_matches=3):
    return_documents = []
    #embeddings = model.encode(df['Cleaned Data'].values.tolist() + [query], convert_to_tensor=True)
    #embeddings = embeddings.cpu().float()
    #cosine_scores = util.pytorch_cos_sim(embeddings[-1], embeddings[:-1])[0]
    cosine_scores = util.pytorch_cos_sim(query_embedding, cleaned_data_embeddings)[0]
    top_results = np.array(cosine_scores).argsort()[::-1][:number_top_matches]
    for idx in top_results:
        a = {
            #"html_ref": "arxiv base url" + str(df.iloc[int(idx)]['Page Number']),
            "Reference": "<a href = https://arxiv.org/pdf/2304.02924v1.pdf>Here is the article</a>",
            "Information": str(df.iloc[int(idx)]['File Name']) + " - Page - " + str(df.iloc[int(idx)]['Page Number']),
            "Title": str(df.iloc[int(idx)]['title'])
        }
        return_documents.append(a) 
    return return_documents

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

 