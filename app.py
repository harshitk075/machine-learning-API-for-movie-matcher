import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle

# load model
df2= pickle.load(open("dataframe.pkl", "rb"))
indices= pickle.load(open("indices.pkl", "rb"))
cosine_sim= pickle.load(open("model.pkl", "rb"))


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


# app
app = Flask(__name__)
CORS(app)

# routes
@app.route('/', methods= ['GET'])
def predict():
    # get data
    data = request.args.get('movie_title')
    res= get_recommendations(data.lower())
    list = res.astype(str).tolist()
    return jsonify({"recommendations": list})  

if __name__ == '__main__':
    app.run(port = 5000, debug=True)