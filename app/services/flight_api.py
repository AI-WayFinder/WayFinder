import os
import requests

from typing import Any, Dict, List
from urllib.parse import urlunparse, urlencode

API_BASE_URL = os.getenv(
    "API_BASE_URL", "localhost:8080"
)  # replace with your real endpoint
API_BASE_SCHEME = os.getenv("API_BASE_SCHEME", "http")  # default to http if not set


class FlightAPIService:
    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        trip_type: str = "oneway",
        return_date: str | None = None,
        max_stops: int = -1,
        max_price: int = 0,
        adults: int = 1,
        children: int = 0,
    ) -> List[Dict[str, Any]]:
        stops = "any"
        if max_stops == 0:
            stops = "nonstop"
        elif max_stops == 1:
            stops = "max1"
        elif max_stops == 2:
            stops = "max2"

        if max_price is None:
            max_price = ""

        query = {
            "date": departure_date,
            "tripType": trip_type,
            "stops": stops,
            "maxPrice": max_price,
            "adults": adults,
            "children": children,
        }

        q = urlencode(query)

        url = urlunparse(
            (
                API_BASE_SCHEME,
                self.base_url,
                f"/flights/{origin}/{destination}",
                "",
                q,
                "",
            )
        )

        response = requests.get(
            url,
            timeout=30,
        )
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API request failed with status code {response.status_code}: {response.text}",
            }

        data = response.json()

        return {
            "flights": data.get("Flights", []),
            "success": True,
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
        }
