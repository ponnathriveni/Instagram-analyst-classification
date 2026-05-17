from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

# Load model files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Input schema
class InputData(BaseModel):
    likes: int
    comments: int
    shares: int
    saves: int
    reach: int
    impressions: int
    media_type: str
    hashtag_count: int
    caption_length: int
    posting_hour: int

# Home route
@app.get("/")
def home():
    return {"message": "API Running Successfully"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):

    # Convert input to dictionary
    input_data = {
        "likes": data.likes,
        "comments": data.comments,
        "shares": data.shares,
        "saves": data.saves,
        "reach": data.reach,
        "impressions": data.impressions,
        "media_type": data.media_type,
        "hashtag_count": data.hashtag_count,
        "caption_length": data.caption_length,
        "posting_hour": data.posting_hour
    }

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in df.columns:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except:
                return {
                    "error": f"Unknown category in column '{col}'"
                }

    # Arrange columns in training order
    df = df[feature_names]

    # Prediction
    prediction = model.predict(df)

    # Decode prediction label
    predicted_label = label_encoder.inverse_transform(prediction)

    return {
        "prediction": predicted_label[0]
    }