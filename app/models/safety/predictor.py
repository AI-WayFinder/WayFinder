from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .v6_features import FEATURE_COLS_V6, SafetyV6FeatureBuilder
from .v9b_best_mlp_config import (
    V9B_MODEL_VERSION,
    V9B_STATE_DICT_PATH,
    V9B_IMPUTER_PATH,
    V9B_SCALER_PATH,
    V9B_FEATURE_COLS,
    V9B_INPUT_DIM,
    V9B_HIDDEN_SIZES,
    V9B_DROPOUT,
    V9B_ACTIVATION,
    V9B_USE_BATCHNORM,
)
from .v9b_model import TorchMLP

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
    Combined predictor that supports:
      - v9b Torch MLP
      - v6 Torch MLP
      - v6 sklearn RF
    """

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.device = torch.device("cpu")

        self.feature_builder = SafetyV6FeatureBuilder()

        self.v6_feature_cols = FEATURE_COLS_V6
        self.v9b_feature_cols = V9B_FEATURE_COLS

        self.v6_scaler = self._load_v6_scaler()
        self.v6_mlp = self._load_v6_mlp_model()
        self.v6_mlp.eval()
        self.v6_rf = self._load_v6_rf_model()

        self.v9b_imputer = self._load_v9b_imputer()
        self.v9b_scaler = self._load_v9b_scaler()
        self.v9b_model = self._load_v9b_mlp_model()
        self.v9b_model.eval()

    # ---------- Loaders ----------

    def _load_v6_scaler(self) -> Any:
        model_path = self.artifacts_dir / "scaler_v6.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"v6 scaler artifact not found at {model_path}")
        return joblib.load(model_path)

    def _load_v6_mlp_model(self) -> nn.Module:
        model_path = self.artifacts_dir / "mlp_v6_best_torch.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"v6 MLP artifact not found at {model_path}")

        loaded = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(loaded, nn.Module):
            return loaded

        if isinstance(loaded, dict):
            state_dict = loaded.get("model_state_dict", loaded.get("state_dict"))
            config: dict[str, Any] = loaded.get("config", {})

            hidden_from_ckpt = config.get("hidden") or config.get("hidden_dims")
            hidden_dims = tuple(hidden_from_ckpt) if hidden_from_ckpt is not None else (128, 128)
            dropout = float(config.get("dropout", 0.2))

            model = MLPRegressorTorch(
                in_dim=len(self.v6_feature_cols),
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            if state_dict is None:
                raise ValueError("v6 checkpoint dict missing model_state_dict/state_dict")
            model.load_state_dict(state_dict)
            return model

        raise ValueError("Unsupported checkpoint format for mlp_v6_best_torch.pt")

    def _load_v6_rf_model(self) -> Any:
        model_path = self.artifacts_dir / "rf_v6.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"v6 RF artifact not found at {model_path}")
        return joblib.load(model_path)

    def _load_v9b_imputer(self) -> Any:
        if not V9B_IMPUTER_PATH.exists():
            raise FileNotFoundError(f"v9b imputer artifact not found at {V9B_IMPUTER_PATH}")
        return joblib.load(V9B_IMPUTER_PATH)

    def _load_v9b_scaler(self) -> Any:
        if not V9B_SCALER_PATH.exists():
            raise FileNotFoundError(f"v9b scaler artifact not found at {V9B_SCALER_PATH}")
        return joblib.load(V9B_SCALER_PATH)

    def _load_v9b_mlp_model(self) -> TorchMLP:
        if not V9B_STATE_DICT_PATH.exists():
            raise FileNotFoundError(f"v9b state_dict artifact not found at {V9B_STATE_DICT_PATH}")

        model = TorchMLP(
            input_dim=V9B_INPUT_DIM,
            hidden_sizes=V9B_HIDDEN_SIZES,
            dropout=V9B_DROPOUT,
            activation=V9B_ACTIVATION,
            use_batchnorm=V9B_USE_BATCHNORM,
        ).to(self.device)

        state_dict = torch.load(V9B_STATE_DICT_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    # ---------- Feature builders ----------

    def _build_all_features(self, latitude: float, longitude: float, country: str | None = None) -> dict[str, Any]:
        return self.feature_builder.build_all_features(
            lat=float(latitude),
            lon=float(longitude),
            country=country,
        )

    def _build_v6_features_df(self, latitude: float, longitude: float, country: str | None = None) -> pd.DataFrame:
        feats = self._build_all_features(latitude, longitude, country)
        missing_cols = [c for c in self.v6_feature_cols if c not in feats]
        if missing_cols:
            raise ValueError(f"Missing expected v6 feature keys: {missing_cols}")
        return pd.DataFrame([[feats[c] for c in self.v6_feature_cols]], columns=self.v6_feature_cols)

    def _build_v9b_features_df(self, latitude: float, longitude: float, country: str | None = None) -> pd.DataFrame:
        feats = self._build_all_features(latitude, longitude, country)
        missing_cols = [c for c in self.v9b_feature_cols if c not in feats]
        if missing_cols:
            raise ValueError(f"Missing expected v9b feature keys: {missing_cols}")
        return pd.DataFrame([[feats[c] for c in self.v9b_feature_cols]], columns=self.v9b_feature_cols)

    def _score_to_risk_band(self, score: float) -> str:
        if score >= 75:
            return "Low risk"
        if score >= 60:
            return "Moderate risk"
        if score >= 45:
            return "Elevated risk"
        return "High risk"

    # ---------- Prediction methods ----------

    def predict_v6(self, latitude: float, longitude: float, country: str | None = None) -> dict[str, Any]:
        X_v6 = self._build_v6_features_df(latitude, longitude, country)
        X_scaled = self.v6_scaler.transform(X_v6)

        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32)
            mlp_score = float(self.v6_mlp(x_tensor).detach().cpu().numpy()[0])

        rf_score = float(self.v6_rf.predict(X_v6)[0])
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
            "feature_count": len(self.v6_feature_cols),
            "features_used": self.v6_feature_cols,
            "input": {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "country": country,
            },
        }

    def predict_v9b(self, latitude: float, longitude: float, country: str | None = None, include_features: bool = False) -> dict[str, Any]:
        X = self._build_v9b_features_df(latitude, longitude, country)
        X_imp = self.v9b_imputer.transform(X)
        X_scaled = self.v9b_scaler.transform(X_imp)

        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32, device=self.device)
            pred = self.v9b_model(x_tensor).detach().cpu().numpy()[0]

        safety_score = float(pred)
        result = {
            "safety_score": safety_score,
            "risk_band": self._score_to_risk_band(safety_score),
            "model_version": V9B_MODEL_VERSION,
            "models_used": ["v9b_torch_mlp"],
            "feature_count": len(self.v9b_feature_cols),
            "features_used": self.v9b_feature_cols,
            "input": {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "country": country,
            },
        }
        if include_features:
            result["features"] = X.iloc[0].to_dict()
        return result

    # ---------- Backward-compatible wrappers ----------

    def predict_score(self, latitude: float, longitude: float, country: str | None = None) -> dict[str, Any]:
        return self.predict_v9b(latitude=latitude, longitude=longitude, country=country, include_features=False)

    def predict_with_features(self, latitude: float, longitude: float, country: str | None = None) -> dict[str, Any]:
        return self.predict_v9b(latitude=latitude, longitude=longitude, country=country, include_features=True)

    def compare_all_models(self, latitude: float, longitude: float, country: str, location_name: str | None = None) -> dict[str, Any]:
        result = {
            "location_name": location_name or "",
            "country": country,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "v9b_score": None,
            "v6_mlp_score": None,
            "v6_rf_score": None,
            "v6_ensemble_score": None,
            "spread_max_min": None,
            "errors": "",
        }

        errors: list[str] = []

        try:
            v9b = self.predict_v9b(latitude=latitude, longitude=longitude, country=country)
            result["v9b_score"] = v9b.get("safety_score")
        except Exception as e:
            errors.append(f"v9b exception: {e}")

        try:
            v6 = self.predict_v6(latitude=latitude, longitude=longitude, country=country)
            result["v6_mlp_score"] = v6.get("mlp_score_v6")
            result["v6_rf_score"] = v6.get("rf_score_v6")
            result["v6_ensemble_score"] = v6.get("safety_score")
        except Exception as e:
            errors.append(f"v6 exception: {e}")

        scores = [
            s for s in [
                result["v9b_score"],
                result["v6_mlp_score"],
                result["v6_rf_score"],
                result["v6_ensemble_score"],
            ]
            if isinstance(s, (int, float))
        ]
        if scores:
            result["spread_max_min"] = round(max(scores) - min(scores), 2)

        result["errors"] = " | ".join(errors)
        return result

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .v6_features import SafetyV6FeatureBuilder
from .v9b_best_mlp_config import (
    V9B_MODEL_VERSION,
    V9B_STATE_DICT_PATH,
    V9B_IMPUTER_PATH,
    V9B_SCALER_PATH,
    V9B_FEATURE_COLS,
    V9B_INPUT_DIM,
    V9B_HIDDEN_SIZES,
    V9B_DROPOUT,
    V9B_ACTIVATION,
    V9B_USE_BATCHNORM,
)
from .v9b_model import TorchMLP


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class SafetyPredictor:
    """
    Production safety predictor using the v9b PyTorch MLP.

    Feature generation:
      - SafetyV6FeatureBuilder builds the full geographic/macroeconomic feature row.
      - We then subset to the exact saved v9b feature list.
      - Saved imputer + scaler are applied before Torch inference.
    """

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.device = torch.device("cpu")

        self.feature_builder = SafetyV6FeatureBuilder()
        self.feature_cols = V9B_FEATURE_COLS

        self.imputer = self._load_imputer()
        self.scaler = self._load_scaler()
        self.model = self._load_mlp_model()
        self.model.eval()

    def _load_imputer(self) -> Any:
        if not V9B_IMPUTER_PATH.exists():
            raise FileNotFoundError(f"Imputer artifact not found at {V9B_IMPUTER_PATH}")
        return joblib.load(V9B_IMPUTER_PATH)

    def _load_scaler(self) -> Any:
        if not V9B_SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {V9B_SCALER_PATH}")
        return joblib.load(V9B_SCALER_PATH)

    def _load_mlp_model(self) -> TorchMLP:
        if not V9B_STATE_DICT_PATH.exists():
            raise FileNotFoundError(f"MLP state_dict artifact not found at {V9B_STATE_DICT_PATH}")

        model = TorchMLP(
            input_dim=V9B_INPUT_DIM,
            hidden_sizes=V9B_HIDDEN_SIZES,
            dropout=V9B_DROPOUT,
            activation=V9B_ACTIVATION,
            use_batchnorm=V9B_USE_BATCHNORM,
        ).to(self.device)

        state_dict = torch.load(V9B_STATE_DICT_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def _build_features_df(
        self,
        latitude: float,
        longitude: float,
        country: str | None = None,
    ) -> pd.DataFrame:
        all_feats = self.feature_builder.build_all_features(
            lat=float(latitude),
            lon=float(longitude),
            country=country,
        )

        missing_cols = [c for c in self.feature_cols if c not in all_feats]
        if missing_cols:
            raise ValueError(
                f"Missing expected feature keys from v9b feature builder: {missing_cols}"
            )

        X = pd.DataFrame(
            [[all_feats[c] for c in self.feature_cols]],
            columns=self.feature_cols,
        )
        return X

    def _score_to_risk_band(self, score: float) -> str:
        if score >= 75:
            return "Low risk"
        if score >= 60:
            return "Moderate risk"
        if score >= 45:
            return "Elevated risk"
        return "High risk"

    def predict_score(
        self,
        latitude: float,
        longitude: float,
        country: str | None = None,
    ) -> dict[str, Any]:
        X = self._build_features_df(
            latitude=latitude,
            longitude=longitude,
            country=country,
        )

        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)

        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32, device=self.device)
            pred = self.model(x_tensor).detach().cpu().numpy()[0]

        safety_score = float(pred)
        risk_band = self._score_to_risk_band(safety_score)

        return {
            "safety_score": safety_score,
            "risk_band": risk_band,
            "model_version": V9B_MODEL_VERSION,
            "models_used": ["v9b_torch_mlp"],
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
        X = self._build_features_df(
            latitude=latitude,
            longitude=longitude,
            country=country,
        )

        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)

        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32, device=self.device)
            pred = self.model(x_tensor).detach().cpu().numpy()[0]

        safety_score = float(pred)
        risk_band = self._score_to_risk_band(safety_score)

        return {
            "safety_score": safety_score,
            "risk_band": risk_band,
            "model_version": V9B_MODEL_VERSION,
            "models_used": ["v9b_torch_mlp"],
            "feature_count": len(self.feature_cols),
            "features_used": self.feature_cols,
            "features": X.iloc[0].to_dict(),
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
    def compare_all_models(self, latitude: float, longitude: float, country: str, location_name: str | None = None) -> dict:
        result = {
            "location_name": location_name or "",
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
            "v9b_score": None,
            "v6_mlp_score": None,
            "v6_rf_score": None,
            "spread_max_min": None,
            "errors": [],
            "v9b_raw": None,
            "v6_raw": None,
        }

        try:
            v9b = self.predict_with_features(
                latitude=latitude,
                longitude=longitude,
                country=country,
            )
            result["v9b_raw"] = v9b

            if isinstance(v9b, dict):
                if "safety_score" in v9b:
                    result["v9b_score"] = v9b.get("safety_score")
                elif "score" in v9b:
                    result["v9b_score"] = v9b.get("score")
                else:
                    result["errors"].append(f"v9b dict missing expected score key: {list(v9b.keys())}")
            else:
                result["errors"].append(f"v9b unexpected return type: {type(v9b).__name__}")
        except Exception as e:
            result["errors"].append(f"v9b exception: {e}")

        try:
            v6 = self.predict_score(
                latitude=latitude,
                longitude=longitude,
                country=country,
            )
            result["v6_raw"] = v6

            if isinstance(v6, dict):
                if "mlp_score_v6" in v6:
                    result["v6_mlp_score"] = v6.get("mlp_score_v6")
                if "rf_score_v6" in v6:
                    result["v6_rf_score"] = v6.get("rf_score_v6")
                if result["v6_mlp_score"] is None and result["v6_rf_score"] is None:
                    result["errors"].append(f"v6 dict missing expected keys: {list(v6.keys())}")
            else:
                result["errors"].append(f"v6 unexpected return type: {type(v6).__name__}")
        except Exception as e:
            result["errors"].append(f"v6 exception: {e}")

        scores = [
            s for s in [
                result["v9b_score"],
                result["v6_mlp_score"],
                result["v6_rf_score"],
            ]
            if isinstance(s, (int, float))
        ]
        if scores:
            result["spread_max_min"] = round(max(scores) - min(scores), 2)

        result["errors"] = " | ".join(result["errors"]) if result["errors"] else ""
        return result
    
    

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

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn

from .feature_pipeline import SafetyFeaturePipeline


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class MLPRegressorTorch(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (128, 128), dropout: float = 0.2) -> None:
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

    Interface is unchanged: predict_score(lat, lon, country) returns
    a dict with per-model scores plus a combined safety_score.
    """

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.pipeline = SafetyFeaturePipeline(self.artifacts_dir)
        self.feature_cols = self.pipeline.load_feature_columns()

        # Load both models
        self.mlp = self._load_mlp_model()
        self.mlp.eval()
        self.rf = self._load_rf_model()

    # ---------- Loading models ----------

    def _load_mlp_model(self) -> nn.Module:
        model_path = self.artifacts_dir / "mlp_v6_best_torch.pt"
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
                raise ValueError("Checkpoint dict does not contain model_state_dict or state_dict.")
            model.load_state_dict(state_dict)
            return model

        raise ValueError("Unsupported model checkpoint format for mlp_v6_best_torch.pt")

    def _load_rf_model(self) -> Any:
        model_path = self.artifacts_dir / "rf_v6.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"RandomForest artifact not found at {model_path}")
        return joblib.load(model_path)

    # ---------- Prediction ----------

    def predict_score(
        self,
        latitude: float,
        longitude: float,
        country: str | None = None,
    ) -> dict[str, Any]:
        """
        Build features via SafetyFeaturePipeline, then:
          - scaled features → MLP v6 (Torch)
          - raw feature cols → RF v6 (sklearn)
        Returns both predictions and a combined safety_score.
        """

        # Build full feature matrix from your existing pipeline
        X_full = self.pipeline.build_features(
            latitude=latitude,
            longitude=longitude,
            country=country,
        )

        # Defensive check: ensure RF sees same columns it was trained on
        missing_cols = [c for c in self.feature_cols if c not in X_full.columns]
        if missing_cols:
            raise ValueError(f"Missing expected feature columns for v6 models: {missing_cols}")

        # RF: unscaled features (as trained in v6_train)
        X_rf = X_full[self.feature_cols]

        # MLP: scaled features (pipeline scaling logic)
        X_scaled = self.pipeline.scale_features(X_full[self.feature_cols])

        # MLP prediction (Torch)
        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X_scaled), dtype=torch.float32)
            mlp_score = float(self.mlp(x_tensor).detach().cpu().numpy()[0])

        # RF prediction (sklearn)
        rf_score = float(self.rf.predict(X_rf)[0])

        # Simple ensemble: mean of both
        combined_score = float((mlp_score + rf_score) / 2.0)

        return {
            "safety_score": combined_score,          # final score your app should use
            "mlp_score_v6": mlp_score,
            "rf_score_v6": rf_score,
            "model_version": "v6_ensemble",
            "models_used": ["mlp_v6_torch", "rf_v6_sklearn"],
            "feature_count": len(self.feature_cols),
            "features_used": self.feature_cols,
        }
    ------------------------------------------------------------
     above this line is current predictor using both MLP and RF
      
        below are other integrations we do not want to part with yet
            


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .feature_pipeline import SafetyFeaturePipeline


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class MLPRegressorTorch(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (128, 128), dropout: float = 0.2) -> None:
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
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.pipeline = SafetyFeaturePipeline(self.artifacts_dir)
        self.feature_cols = self.pipeline.load_feature_columns()
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> nn.Module:
        model_path = self.artifacts_dir / "mlp_v6_best_torch.pt"

        loaded = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(loaded, nn.Module):
            return loaded

        if isinstance(loaded, dict):
            state_dict = loaded.get("model_state_dict", loaded.get("state_dict"))
            config: dict[str, Any] = loaded.get("config", {})

            # v6_128_128 stores config["hidden"] and config["dropout"]
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
                raise ValueError("Checkpoint dict does not contain model_state_dict or state_dict.")
            model.load_state_dict(state_dict)
            return model

        raise ValueError("Unsupported model checkpoint format for mlp_v6_best_torch.pt")

    def predict_score(
        self,
        latitude: float,
        longitude: float,
        country: str | None = None,
    ) -> dict[str, Any]:
        X = self.pipeline.build_features(
            latitude=latitude,
            longitude=longitude,
            country=country,
        )
        X_scaled = self.pipeline.scale_features(X)

        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            score = float(self.model(x_tensor).detach().cpu().numpy()[0])

        return {
            "safety_score": score,
            "model_version": "v6",
            "feature_count": len(self.feature_cols),
            "features_used": self.feature_cols,
        }

"""
If mlp_v6_best_torch.pt is just a raw state dict from a different training class, you will need to match the exact hidden layer sizes from training. That is normal for PyTorch inference unless you exported a full model or TorchScript artifact. Defining a dedicated inference model that mirrors training is a common approach.


"""

# app/models/safety/predictor.py
import joblib
import pandas as pd

from .v6_config import MLP_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH
from .v6_features import FEATURE_COLS_V6


class SafetyV6Predictor:
    def __init__(self):
        self.mlp = joblib.load(MLP_MODEL_PATH)
        self.rf = joblib.load(RF_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def _check_cols(self, df: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLS_V6 if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features for safety v6 model: {missing}")

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df must already have FEATURE_COLS_V6 (e.g. from feature_pipeline).
        Returns df with mlp_pred_v6 and rf_pred_v6 appended.
        """
        self._check_cols(df)
        X = df[FEATURE_COLS_V6].copy()
        X_scaled = self.scaler.transform(X)

        df = df.copy()
        df["mlp_pred_v6"] = self.mlp.predict(X_scaled)
        df["rf_pred_v6"] = self.rf.predict(X)
        return df

    def predict_row(self, row: pd.Series) -> dict:
        missing = [c for c in FEATURE_COLS_V6 if c not in row.index]
        if missing:
            raise ValueError(f"Missing features in row: {missing}")

        x = row[FEATURE_COLS_V6].to_frame().T
        x_scaled = self.scaler.transform(x)

        return {
            "mlp_pred_v6": float(self.mlp.predict(x_scaled)[0]),
            "rf_pred_v6": float(self.rf.predict(x)[0]),
        }

        '''