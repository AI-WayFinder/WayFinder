from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
COMPILED_DIR = DATA_DIR / "compiled_model_ready"

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
V9B_DIR = ARTIFACTS_DIR / "geo_safety_model_v9b_torch_mlp"

TARGET_COL = "safety_index"

V9B_STATE_DICT_PATH = V9B_DIR / "v9b_best_mlp_state_dict.pt"
V9B_IMPUTER_PATH = V9B_DIR / "v9b_best_mlp_imputer.joblib"
V9B_SCALER_PATH = V9B_DIR / "v9b_best_mlp_scaler.joblib"
V9B_FEATURES_PATH = V9B_DIR / "v9b_best_mlp_features.joblib"

for path in [V9B_STATE_DICT_PATH, V9B_IMPUTER_PATH, V9B_SCALER_PATH, V9B_FEATURES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing v9b artifact: {path}")

V9B_MODEL_VERSION = "v9b_torch_mlp"

V9B_HIDDEN_SIZES = (512, 256, 128)
V9B_DROPOUT = 0.2
V9B_ACTIVATION = "gelu"
V9B_USE_BATCHNORM = True

def load_v9b_feature_columns() -> list[str]:
    cols = joblib.load(V9B_FEATURES_PATH)
    return [str(c) for c in cols]

V9B_FEATURE_COLS = load_v9b_feature_columns()
V9B_INPUT_DIM = len(V9B_FEATURE_COLS)