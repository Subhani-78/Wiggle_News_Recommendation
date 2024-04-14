from typing import List
from src.newsrec import history_dataset, news_dataset, euclidean_distance_based_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

def get_recommendations(user_id: str) -> List[str]:
    user_click_history = history_dataset[history_dataset.userId == user_id].reset_index()['click_history'][0]
    combined_similar_titles = {}  

    for history in user_click_history:
        try:
            title = news_dataset[news_dataset['News ID'] == history]['Title'].values[0]
            input_title = title
            similar_titles = euclidean_distance_based_model(input_title, 6, news_dataset)
            if similar_titles is not None:
                combined_similar_titles.update(similar_titles)
        except IndexError:
            pass

    recommendations = list(combined_similar_titles.keys())
    return recommendations

# print(get_recommendations("U13740"))

class Query(BaseModel):
    user_id: str

@app.post("/wiggle_endpoint")
async def wiggle_endpoint(query: Query):
    user_id = query.user_id
    recommendations = get_recommendations(user_id)
    return recommendations
