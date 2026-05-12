from fastapi import FastAPI ,Query
import requests

app = FastAPI()

API_KEY = "AIzaSyBYPb5NoBa4OgaNrKO_MTYKAfc6ifeKazI"

@app.get("/distance")
def get_distance(
    lat1: float = Query(..., description="origin latitude"),
    lng1: float = Query(..., description="origin longitude"),
    lat2: float = Query(..., description="destination latitude"),
    lng2: float = Query(..., description="destination longitude")
):

    url = "https://maps.googleapis.com/maps/api/distancematrix/json" # this is a end point of google server here Ye endpoint hai Google kaYe service distance calculate karti hai

    params = {
        "origins": f"{lat1},{lng1}",
        "destinations": f"{lat2},{lng2}",
        "mode": "driving",
        "key": API_KEY
    }

    response = requests.get(url, params=params)

    return response.json()