import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import pickle

# 31-05-2023
def dataPreProcess(data):
    df = pd.read_json(data)
    df.category = df.category.apply(lambda y: np.nan if len(y) == 0 else y)
    df.brand = df.brand.apply(lambda y: np.nan if len(y) == 0 else y)
    df.selectedcategory = df.selectedcategory.fillna('')
    df.gender =  df.gender.fillna('')
    df.selectedcategory = df.selectedcategory.apply(lambda y: np.nan if len(y) == 0 else y)
    df['id'] = df['_id'].apply(lambda x: x['$oid'])
    df = df[['id','gender' ,'email','firstname', 'lastname', 'profileimage', 'COO', 'category', 'brand', 'selectedcategory']]
    df = df.fillna(method="pad")
    df['tag'] = df['COO'] + df['category'] + df['brand'] + df['selectedcategory']
    df['tag'] = df['tag'].astype(str)
    return df[['id','firstname', 'lastname', 'profileimage', 'email', 'gender','tag']]

def recommendFriend(email,limit,page):
    limit = int(limit)
    page = int(page)
    start = (page * limit) -limit
    df_result = dataPreProcess('jessuusers.json')
    if df_result.empty:
        print("No data available.")
        return []

    new_df = df_result[['id', 'firstname', 'lastname', 'profileimage', 'email','gender', 'tag']]
    new_df = new_df.dropna(subset=['tag'])
    new_df['tag'] = new_df['tag'].astype(str)

    tfv = TfidfVectorizer()
    tfv_matrix = tfv.fit_transform(df_result['tag'])

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tag']).toarray()

    similarity = cosine_similarity(vectors)

    if email not in new_df['email'].values:
        print(f"No data available for email: {email}")
        return []

    email_index = new_df[new_df['email'] == email].index[0]
    distances = similarity[email_index]
    email_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[start:limit]
    

    e_mail= []
    for i in email_list:
        e_mail.append(new_df.iloc[i[0]].email)
    e = list(set(e_mail))
    df = new_df.loc[new_df['email'].isin(e)]
    df = df.drop_duplicates(subset=['email']) 
    fname = df['firstname'].tolist()
    lname = df['lastname'].tolist()
    email = df['email'].tolist()
    gender = df['gender'].tolist()
    profileimage = df['profileimage'].tolist()
    recommendedData = [{'firstname': k, 'lastname': v,'friend_email':i,"profileImage":j,'gender':g} for k, v,i,j,g in zip(fname,lname,email,profileimage,gender)]

    return  recommendedData


flask_app = Flask(__name__)
with open('recommendFriend.pkl', 'rb') as file:
    model = pickle.load(file)

@flask_app.route("/")
def hello():
    pickle.dump(recommendFriend, open("recommendFriend.pkl", "wb"))
    return "Hello, World!.."

@flask_app.route("/predict", methods=["GET"])
def predict():
    json_ = request.args.get('email')
    limit = request.args.get('limit')
    page = request.args.get('page')
    prediction = model(json_,limit,page)
    if prediction is not None:
        return jsonify(prediction)
    else:
        return ("No friend Found")


if __name__ == "__main__":
    # flask_app.run(debug=True, use_reloader=False)
    flask_app.run(host='0.0.0.0', port=80)
