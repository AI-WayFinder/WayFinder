from __future__ import annotations

import pandas as pd
import streamlit as st

from components.location_picker import location_picker
from models.safety.predictor import SafetyPredictor
from models.safety.schemas import SafetyRequest
from services.memory_service import MemoryService
from services.model_service import ModelService
from services.safety_service import SafetyService
from ui.chat_handlers import handle_assistant_response, handle_user_message
from ui.styles import inject_global_styles


@st.cache_resource
def get_model_service(_cache_version: str = "v6-mps-eager-attn") -> ModelService:
    return ModelService()


@st.cache_resource
def get_safety_service() -> SafetyService:
    return SafetyService()


def get_selected_location_fields() -> dict:
    selected = st.session_state.get("selected_location")
    if not selected:
        return {
            "lat": None,
            "lon": None,
            "country": None,
            "location_name": None,
        }

    short_location_name = (
        selected.get("city")
        or selected.get("county")
        or selected.get("state_region")
        or selected.get("country")
    )

    return {
        "lat": selected.get("lat"),
        "lon": selected.get("lon"),
        "country": selected.get("country"),
        "location_name": short_location_name,
    }


FIXED_TEST_POINTS = [
    {"location_name": "San Diego, CA", "country": "United States", "latitude": 32.7157, "longitude": -117.1611},
    {"location_name": "New York, NY", "country": "United States", "latitude": 40.7128, "longitude": -74.0060},
    {"location_name": "Detroit, MI", "country": "United States", "latitude": 42.3314, "longitude": -83.0458},
    {"location_name": "Salt Lake City, UT", "country": "United States", "latitude": 40.7608, "longitude": -111.8910},
    {"location_name": "Boulder, CO", "country": "United States", "latitude": 40.0150, "longitude": -105.2705},
    {"location_name": "London", "country": "United Kingdom", "latitude": 51.5074, "longitude": -0.1278},
    {"location_name": "Cape Town", "country": "South Africa", "latitude": -33.9249, "longitude": 18.4241},
    {"location_name": "Tokyo", "country": "Japan", "latitude": 35.6762, "longitude": 139.6503},
    {"location_name": "Mexico City", "country": "Mexico", "latitude": 19.4326, "longitude": -99.1332},
    {"location_name": "Reykjavik", "country": "Iceland", "latitude": 64.1466, "longitude": -21.9426},
]


SMALL_TOWN_TEST_POINTS = [
    {"location_name": "Moab, UT", "country": "United States", "latitude": 38.5733, "longitude": -109.5498},
    {"location_name": "Kanab, UT", "country": "United States", "latitude": 37.0475, "longitude": -112.5263},
    {"location_name": "Marfa, TX", "country": "United States", "latitude": 30.3094, "longitude": -104.0206},
    {"location_name": "Cody, WY", "country": "United States", "latitude": 44.5263, "longitude": -109.0565},
    {"location_name": "Bishop, CA", "country": "United States", "latitude": 37.3635, "longitude": -118.3951},
    {"location_name": "Taos, NM", "country": "United States", "latitude": 36.4072, "longitude": -105.5731},
    {"location_name": "Sedona, AZ", "country": "United States", "latitude": 34.8697, "longitude": -111.7610},
    {"location_name": "Ajo, AZ", "country": "United States", "latitude": 32.3717, "longitude": -112.8607},
    {"location_name": "Terlingua, TX", "country": "United States", "latitude": 29.3291, "longitude": -103.5602},
    {"location_name": "Escalante, UT", "country": "United States", "latitude": 37.7700, "longitude": -111.6027},
]


def run_model_comparison_batch(predictor, points: list[dict]) -> pd.DataFrame:
    rows = []
    for p in points:
        row = predictor.compare_all_models(
            latitude=p["latitude"],
            longitude=p["longitude"],
            country=p["country"],
            location_name=p["location_name"],
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    score_cols = ["v9b_score", "v6_mlp_score", "v6_rf_score", "spread_max_min"]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    return df


def init_chat_page_state() -> None:
    defaults = {
        "selected_location": None,
        "safety_result": None,
        "safety_debug": None,
        "comparison_fixed_df": None,
        "comparison_small_town_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_chat_page() -> None:
    init_chat_page_state()

    st.title("Travel Agent AI")
    inject_global_styles()

    MemoryService.initialize()
    model_service = get_model_service()
    safety_service = get_safety_service()

    with st.expander("Pick a location on the map", expanded=False):
        picked_location = location_picker(
            key="wayfinder_location_picker",
            height=760,
            default=st.session_state["selected_location"],
        )

        if picked_location:
            st.session_state["selected_location"] = picked_location
            st.session_state["safety_result"] = None

    if st.session_state["selected_location"]:
        selected = st.session_state["selected_location"]
        fields = get_selected_location_fields()

        st.subheader("Selected location")
        st.write(
            {
                "lat": selected.get("lat"),
                "lon": selected.get("lon"),
                "country": selected.get("country"),
                "state_region": selected.get("state_region"),
                "county": selected.get("county"),
                "city": selected.get("city"),
            }
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Latitude", f"{fields['lat']:.6f}" if fields["lat"] is not None else "—")
        col2.metric("Longitude", f"{fields['lon']:.6f}" if fields["lon"] is not None else "—")
        col3.metric("Country", fields["country"] if fields["country"] else "—")

        can_score = fields["lat"] is not None and fields["lon"] is not None

        if st.button("Run safety score", disabled=not can_score):
            st.session_state["safety_debug"] = "button_clicked"
            try:
                req = SafetyRequest(
                    latitude=float(fields["lat"]),
                    longitude=float(fields["lon"]),
                    country=fields["country"],
                    location_name=fields["location_name"],
                )

                st.session_state["safety_debug"] = {
                    "stage": "request_built",
                    "request": {
                        "latitude": req.latitude,
                        "longitude": req.longitude,
                        "country": req.country,
                        "location_name": req.location_name,
                    },
                }

                result = safety_service.assess_request(
                    req,
                    include_details=True,
                )

                st.session_state["safety_result"] = result
                st.session_state["safety_debug"] = {
                    "stage": "result_returned",
                    "result": result,
                }
                st.success("Safety score completed.")

            except Exception as e:
                st.session_state["safety_debug"] = {
                    "stage": "exception",
                    "error": repr(e),
                }

    if st.session_state["safety_result"] is not None:
        result = st.session_state["safety_result"]

        st.subheader("Safety result")

        if result.get("success"):
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Safety score",
                f"{result['safety_score']:.2f}" if result.get("safety_score") is not None else "—",
            )
            c2.metric("Risk band", result.get("risk_band") or "—")
            c3.metric("Model", result.get("model_version") or "—")

            with st.expander("Prediction details", expanded=False):
                st.json(result)
        else:
            st.error(f"Safety scoring failed: {result.get('error')}")

    if st.session_state.get("safety_debug") is not None:
        with st.expander("Safety debug", expanded=True):
            st.json(st.session_state["safety_debug"])

    for message in MemoryService.get_display_messages():
        with st.chat_message(message.role):
            st.markdown(message.content)

    if st.button("Clear chat"):
        MemoryService.clear()
        st.rerun()

    user_input = st.chat_input("Ask about routes, destinations, or itineraries...")

    if user_input:
        handle_user_message(user_input)
        handle_assistant_response(model_service)

    st.markdown("## Model comparison test harness")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run 10 fixed-city comparison"):
            predictor = SafetyPredictor()
            st.session_state["comparison_fixed_df"] = run_model_comparison_batch(
                predictor,
                FIXED_TEST_POINTS,
            )

    with col2:
        if st.button("Run 10 small-town comparison"):
            predictor = SafetyPredictor()
            st.session_state["comparison_small_town_df"] = run_model_comparison_batch(
                predictor,
                SMALL_TOWN_TEST_POINTS,
            )

    if st.session_state.get("comparison_fixed_df") is not None:
        st.markdown("### Fixed-city results")
        st.dataframe(
            st.session_state["comparison_fixed_df"],
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.get("comparison_small_town_df") is not None:
        st.markdown("### Small-town results")
        st.dataframe(
            st.session_state["comparison_small_town_df"],
            use_container_width=True,
            hide_index=True,
        )