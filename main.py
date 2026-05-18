import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load single bundle
data = joblib.load("bundle.pkl")

model = data["model"]
label_encoder = data["label_encoder"]
encoders = data["encoders"]
feature_names = data["feature_names"]

class InputData(BaseModel):
    likes: int
    comments: int
    shares: int
    saves: int
    reach: int
    impressions: int
    media_type: str
    hashtags_count: int
    caption_length: int
    post_hour: int
    follower_count: int
    engagement_rate: float
    account_type: str
    content_category: str
    has_call_to_action: int
    followers_gained: int


@app.post("/")
def predict(data_input: InputData):

    df = pd.DataFrame([data_input.dict()])

    for col in encoders:
        df[col] = encoders[col].transform(df[col].astype(str))

    df = df[feature_names]

    prediction = model.predict(df)
    label = label_encoder.inverse_transform(prediction)

    return {"prediction": label[0]}
