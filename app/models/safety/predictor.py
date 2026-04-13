"""Ensemble safety predictor combining a PyTorch MLP and a scikit-learn Random Forest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .v6_features import FEATURE_COLS_V6, SafetyV6FeatureBuilder


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class MLPRegressorTorch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SafetyPredictor:
    """
    Uses both:
      - Torch MLP v6 checkpoint: mlp_v6_best_torch.pt
      - sklearn RF v6: rf_v6.pkl

    Feature generation:
      - SafetyV6FeatureBuilder builds the full v6 feature row
      - StandardScaler artifact is applied only for the Torch MLP
      - RF uses raw features, matching v5/v6 training semantics

    Returns:
      - mlp_score_v6
      - rf_score_v6
      - safety_score (ensemble)
      - metadata the agent can use when crafting responses
    """

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR

        self.feature_builder = SafetyV6FeatureBuilder()
        self.feature_cols = FEATURE_COLS_V6

        self.scaler = self._load_scaler()
        self.mlp = self._load_mlp_model()
        self.mlp.eval()
        self.rf = self._load_rf_model()

    # ---------- Loading models ----------

    def _load_scaler(self) -> Any:
        model_path = self.artifacts_dir / "scaler_v6.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {model_path}")
        return joblib.load(model_path)

    def _load_mlp_model(self) -> nn.Module:
        model_path = self.artifacts_dir / "mlp_v6_best_torch.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"MLP artifact not found at {model_path}")

        loaded = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(loaded, nn.Module):
            return loaded

        if isinstance(loaded, dict):
            state_dict = loaded.get("model_state_dict", loaded.get("state_dict"))
            config: dict[str, Any] = loaded.get("config", {})

            hidden_from_ckpt = config.get("hidden") or config.get("hidden_dims")
            if hidden_from_ckpt is None:
                hidden_dims = (128, 128)
            else:
                hidden_dims = tuple(hidden_from_ckpt)

            dropout = float(config.get("dropout", 0.2))

            model = MLPRegressorTorch(
                in_dim=len(self.feature_cols),
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            if state_dict is None:
                raise ValueError(
                    "Checkpoint dict does not contain model_state_dict or state_dict."
                )
            model.load_state_dict(state_dict)
            return model

        raise ValueError("Unsupported model checkpoint format for mlp_v6_best_torch.pt")

    def _load_rf_model(self) -> Any:
        model_path = self.artifacts_dir / "rf_v6.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"RandomForest artifact not found at {model_path}")
        return joblib.load(model_path)

    # ---------- Feature helpers ----------

    def _build_features_df(
        self,
        latitude: float,
        longitude: float,
        country: str | None = None,
    ) -> pd.DataFrame:
        feats = self.feature_builder.build_all_features(
            lat=float(latitude),
            lon=float(longitude),
            country=country,
        )

        missing_cols = [c for c in self.feature_cols if c not in feats]
        if missing_cols:
            raise ValueError(f"Missing expected feature keys from v6 feature builder: {missing_cols}")

        X = pd.DataFrame(
            [[feats[c] for c in self.feature_cols]],
            columns=self.feature_cols,
        )
        return X

    # ---------- Prediction ----------

    def predict_score(
        self,
        latitude: float,
        longitude: float,
        country: str | None = None,
    ) -> dict[str, Any]:
        """
        Build features via SafetyV6FeatureBuilder, then:
          - scaled features -> MLP v6 (Torch)
          - raw features -> RF v6 (sklearn)

        Returns both predictions plus an ensemble safety_score.
        """

        # Build feature row in exact trained column order
        X_rf = self._build_features_df(
            latitude=latitude,
            longitude=longitude,
            country=country,
        )

        # MLP uses scaled features, matching training workflow
        X_scaled = self.scaler.transform(X_rf)

        # Torch MLP prediction
        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32)
            mlp_score = float(self.mlp(x_tensor).detach().cpu().numpy()[0])

        # RF prediction
        rf_score = float(self.rf.predict(X_rf)[0])

        # Ensemble: simple average for now
        combined_score = float((mlp_score + rf_score) / 2.0)

        # Helpful disagreement metric for agent-side response logic
        model_spread = float(abs(mlp_score - rf_score))

        # Optional confidence heuristic from agreement
        if model_spread < 3.0:
            agreement_band = "high"
        elif model_spread < 7.0:
            agreement_band = "medium"
        else:
            agreement_band = "low"

        return {
            "safety_score": combined_score,
            "mlp_score_v6": mlp_score,
            "rf_score_v6": rf_score,
            "model_spread": model_spread,
            "agreement_band": agreement_band,
            "model_version": "v6_ensemble",
            "models_used": ["mlp_v6_torch", "rf_v6_sklearn"],
            "feature_count": len(self.feature_cols),
            "features_used": self.feature_cols,
            "input": {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "country": country,
            },
        }

    def predict_with_features(
        self,
        latitude: float,
        longitude: float,
        country: str | None = None,
    ) -> dict[str, Any]:
        """
        Same as predict_score, but also returns the built feature values.
        Useful for debugging, agent explanations, and inspecting derived signals.
        """
        X_rf = self._build_features_df(
            latitude=latitude,
            longitude=longitude,
            country=country,
        )

        X_scaled = self.scaler.transform(X_rf)

        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32)
            mlp_score = float(self.mlp(x_tensor).detach().cpu().numpy()[0])

        rf_score = float(self.rf.predict(X_rf)[0])
        combined_score = float((mlp_score + rf_score) / 2.0)
        model_spread = float(abs(mlp_score - rf_score))

        if model_spread < 3.0:
            agreement_band = "high"
        elif model_spread < 7.0:
            agreement_band = "medium"
        else:
            agreement_band = "low"

        return {
            "safety_score": combined_score,
            "mlp_score_v6": mlp_score,
            "rf_score_v6": rf_score,
            "model_spread": model_spread,
            "agreement_band": agreement_band,
            "model_version": "v6_ensemble",
            "models_used": ["mlp_v6_torch", "rf_v6_sklearn"],
            "feature_count": len(self.feature_cols),
            "features_used": self.feature_cols,
            "features": X_rf.iloc[0].to_dict(),
            "input": {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "country": country,
            },
        }

    def predict_batch(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Batch prediction helper for multiple user queries.

        Each row should look like:
            {"latitude": 34.05, "longitude": -118.24, "country": "United States"}
        """
        outputs: list[dict[str, Any]] = []
        for row in rows:
            outputs.append(
                self.predict_score(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    country=row.get("country"),
                )
            )
        return outputs