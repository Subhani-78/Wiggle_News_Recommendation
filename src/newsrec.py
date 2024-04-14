import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
import ast

headline_vectorizer = CountVectorizer()

news_dataset = pd.read_csv("data\\news_dataset.csv")
news_dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
news_dataset.reset_index(drop=True, inplace=True)

history_dataset = pd.read_csv("data\click_history.csv")
history_dataset.drop(['Unnamed: 0'], axis = 1, inplace = True)
history_dataset.reset_index(drop=True, inplace=True)
history_dataset['click_history'] = history_dataset['click_history'].fillna('[]')
history_dataset['click_history'] = history_dataset['click_history'].apply(ast.literal_eval)

def euclidean_distance_based_model(title, num_similar_items, data):
    try:
        row_index = data[data['Title'] == title].index[0]
        category = data['Category'][row_index]
        category_data = data[data['Category'] == category]
        headline_features = headline_vectorizer.fit_transform(category_data['Title'].values)
        couple_dist = pairwise_distances(headline_features, headline_features[row_index])
        indices = np.argsort(couple_dist.ravel())[1:num_similar_items + 1] 
        
        similar_titles_dict = {}
        for idx in indices:
            similar_titles_dict[category_data.iloc[idx]['News ID']] = category_data.iloc[idx]['Title']

        return similar_titles_dict
    except:
        return None

