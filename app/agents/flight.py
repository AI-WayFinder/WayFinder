from services.intent import IntentService
from tools.flight_search import FlightSearchTool

import json


class FlightAgent:
    def __init__(self) -> None:
        self.intent_service = IntentService()
        self.flight_tool = FlightSearchTool()

    def handle(self, user_message: str) -> str:
        request = self.intent_service.extract_flight_request(user_message)
        result = self.flight_tool.run(request)

        if not result["data"]["success"]:
            return f"It seems like there was an error fetching the flight data: {result['data']['error']}."

        data = result["data"]

        if not data["flights"]:
            return "I found no flights matching those filters."

        return summarize_flights_for_chat(data)


def normalize_flight(raw: dict) -> dict:
    return {
        "is_top": raw.get("IsTop", False),
        "airline_name": raw.get("Airline", {}).get("Name"),
        "operated_by": raw.get("Airline", {}).get("OperatedBy"),
        "departure_time": raw.get("Departure"),
        "arrival_time": raw.get("Arrival"),
        "arrival_time_ahead": raw.get("ArrivalTimeAhead"),
        "duration": raw.get("Duration"),
        "stops": raw.get("Stops"),
        "delay": raw.get("Delay"),
        "price": raw.get("Price"),
        "flight_number": raw.get("Number"),
        "emissions_current": raw.get("Emissions", {}).get("Current"),
        "emissions_typical": raw.get("Emissions", {}).get("Typical"),
        "emissions_savings": raw.get("Emissions", {}).get("Savings"),
        "emissions_percentage_diff": raw.get("Emissions", {}).get("PercentageDiff"),
        "environmental_ranking": raw.get("Emissions", {}).get("EnvironmentalRanking"),
        "contrails_impact": raw.get("Emissions", {}).get("ContrailsImpact"),
        "raw": raw,
    }


def format_flight_for_chat(flight: dict) -> str:
    airline = flight.get("airline_name", "Unknown airline")
    departure = flight.get("departure_time", "Unknown departure")
    arrival = flight.get("arrival_time", "Unknown arrival")
    duration = flight.get("duration", "Unknown duration")
    stops = flight.get("stops", "Unknown")
    price = flight.get("price", "Unknown price")

    stop_text = (
        "nonstop" if stops == 0 else f"{stops} stop" if stops == 1 else f"{stops} stops"
    )

    return f"{airline} | {departure} to {arrival} | {duration} | {stop_text} | {price}"


def summarize_flights_for_chat(data: dict) -> str:
    normalized = [normalize_flight(f) for f in data["flights"]]

    if not normalized:
        return "I couldn’t find any matching flights."

    lines = [
        "Here are a the best flight options for {date} I found that are leaving from {origin} and arriving at {destination}:\n".format(
            date=data["departure_date"],
            origin=data["origin"],
            destination=data["destination"],
        )
    ]

    num = 1
    for flight in normalized:
        if flight["is_top"]:
            prefix = "⭐ " if flight["is_top"] else ""
            lines.append(f"{num}. {prefix}{format_flight_for_chat(flight)}")
            num += 1

    return "\n".join(lines)
