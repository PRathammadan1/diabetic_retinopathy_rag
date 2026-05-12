from fastapi import FastAPI ,Query
import requests

app = FastAPI()

API_KEY = "AIzaSyBYPb5NoBa4OgaNrKO_MTYKAfc6ifeKazI"

@app.get("/places")
def get_places(lat: float =Query(...,description="in this, we get a loaction of latitude"), 
               lng: float =Query(..., description="longitude"), 
               place_type: str = "hospital",
               keyword: str="ophthalmologist near me"):

    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location"
    }

    body = {
        "textQuery": keyword,
        "maxResultCount": 10,
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lng
                },
                "radius": 3000.0
            }
        }
    }

    response = requests.post(url, json=body, headers=headers)
    return response.json()