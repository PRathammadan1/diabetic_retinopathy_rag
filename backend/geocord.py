from fastapi import FastAPI , Query
import requests

app = FastAPI()

API_KEY = ""


@app.get("/geocode")
def geocode(address: str = Query(..., description="give address", example="delhi")):
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {
        "address": address,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()