# WayFinder

An AI-powered travel planning assistant that combines a local large language model with real-time flight search and ML-based destination safety scoring. Users can search flights through natural language conversation, explore destinations on an interactive map, and get safety assessments powered by an ensemble of neural network and random forest models.

## Features

- **Conversational flight search** — Ask for flights in plain English. The agent resolves city names to airport codes, validates dates, and returns real results from a flight API. Strict guardrails prevent the LLM from hallucinating flight data.
- **Multi-airport flight search** — One query fans out across all airports serving the destination metro and renders results grouped by airport with airline, duration, stops, and price.
- **Safety scoring** — Select any location on an interactive map and get a safety score (0-100) with a risk band (low / moderate / elevated / high). Predictions are made by an ensemble of a PyTorch MLP and a scikit-learn Random Forest trained on 45 features including KNN neighborhood crime/safety indices, population density, and country-level macro indicators.
- **Per-destination safety dial** — Each airport card shows a compact numbered gauge (0-100) with risk band, pulled from a KNN-based city safety model.
- **Deterministic safety path** — Asking "Is Paris safe?" or "Safety Vancouver" calls the safety assessment tool directly, so results are consistent regardless of phrasing or capitalization.
- **Robust location resolution** — Multi-word and qualified queries like "Vancouver, BC" or "vancouver canada" fall through progressively shorter prefixes until the geocoder and airport search both find a match.
- **Local LLM** — Runs Qwen 2.5 1.5B Instruct locally with tool-calling support. No API keys or cloud dependencies for the language model. Supports CUDA, Apple Silicon (MPS), and CPU.
- **Interactive map** — Leaflet.js-based location picker built as a custom Streamlit component for selecting destinations and triggering safety assessments.
- **Token-aware context trimming** — The LLM thread is pre-trimmed to fit the model's input budget, dropping oldest tool results first so long conversations stay responsive.

## Safety Score Model

WayFinder ships with a custom ML-based safety scoring model that predicts a continuous safety score for any city or geographic point. The score is surfaced both as a standalone safety assessment and as the per-airport dial shown on flight results.

### How it works

The core is a feedforward Multilayer Perceptron (MLP) regression model implemented in PyTorch. The production architecture uses three fully connected hidden layers (128 -> 64 -> 32) with ReLU activations, dropout, and L2 weight decay to keep the model honest against a relatively small ~500-row labeled city dataset. It's trained with MSE loss and Adam on an 80/20 hold-out split, with early stopping based on validation RMSE.

At inference, WayFinder actually runs two independently trained variants in parallel -- a crime-aware model (uses city-level crime and safety indices where available) and a crime-agnostic model (geographic and macro features only). Comparing the two acts as a built-in cross-check and gracefully degrades when a queried point falls outside the labeled city catalog.

### Model features

The feature vector for a given location combines several broad groups:

- **City-level crime and safety indices** -- Numbeo-style crime and perceived-safety scores for labeled cities. The target city's own crime index is strictly excluded during training to prevent target leakage.
- **KNN neighborhood aggregates** -- Crime and safety averages computed over the nearest labeled cities (weighted and unweighted k=5 / k=10), plus distance-to-nearest-labeled-city features. This is what lets the model score unseen locations by interpolating from labeled neighbors.
- **Density & gravity features** -- Log-transformed population counts, population gravity, and city counts within 50 / 100 / 250 km radii.
- **Country-level macro indicators** -- GDP, GDP per capita, unemployment, homicide rate, life expectancy, and governance signals (rule of law, political stability, press freedom, Global Peace Index).
- **Geographic base features** -- Latitude, longitude, and administrative country identifiers.

Data is sourced from open global datasets including the World Bank (socioeconomic and homicide data), UNODC Global Study on Homicide, the Global Peace Index, and Reporters Without Borders' World Press Freedom Index.

### Handling unseen cities

Most real-world queries don't land on a perfectly labeled city. For any point on Earth, the feature pipeline geocodes the query, finds the nearest labeled cities via KNN, and computes neighborhood aggregates plus macro context for the surrounding region. If city-level crime data is available for the queried point the crime-aware model runs at full fidelity; otherwise the score falls back to the crime-agnostic regime, and the returned payload flags the confidence accordingly.

### Outputs

Each safety assessment returns:

- **`safety_score`** -- A continuous 0-100 value (higher is safer).
- **`risk_band`** -- A bucketed label derived from the score:
  - `low` (75+)
  - `moderate` (55-74)
  - `elevated` (35-54)
  - `high` (<35)
- **Factor breakdown** -- The most influential city-specific signals behind the score, including neighborhood crime / safety averages and the nearest labeled city's own values, used to explain the result conversationally in chat.
- **Confidence indicator** -- Whether the crime-aware model ran with full feature availability or fell back to the crime-agnostic regime.

In the chat UI these outputs are rendered as a conversational markdown response for standalone safety queries, and as a compact numbered dial with a pointer and risk band label on each airport's flight card.

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit, Leaflet.js (custom component) |
| LLM | Qwen 2.5 1.5B Instruct (HuggingFace Transformers) |
| ML Models | PyTorch (MLP), scikit-learn (Random Forest), joblib |
| Data | Pandas, NumPy |
| Flight API | Docker container (scraper service), Requests |
| Environment | Conda (Python 3.12.8) |

## Project Structure

```
WayFinder/
├── app/
│   ├── main.py                          # Streamlit entry point
│   ├── core/
│   │   └── config.py                    # App settings (model name, tokens, temperature)
│   ├── ui/
│   │   ├── chat_page.py                 # Main page: map, safety panel, chat
│   │   ├── chat_handlers.py             # User/assistant message handling
│   │   ├── renderers.py                 # Streaming response rendering
│   │   └── styles.py                    # Global CSS
│   ├── agents/
│   │   ├── local_tool_agent.py          # Agent orchestrator (tool loop + streaming)
│   │   ├── tool_executor.py             # Executes tools (flights, airports, safety)
│   │   ├── tool_call_parser.py          # Parses Qwen <tool_call> blocks
│   │   └── tool_definitions.py         # OpenAI-style tool schemas
│   ├── models/
│   │   ├── chat.py                      # ChatMessage dataclass
│   │   ├── flight_search.py             # FlightSearchRequest dataclass
│   │   └── safety/
│   │       ├── schemas.py               # SafetyRequest / SafetyResult
│   │       ├── predictor.py             # Ensemble predictor (MLP + RF)
│   │       ├── v6_features.py           # Feature engineering (45 features)
│   │       └── artifacts/               # Trained model weights & scaler
│   ├── services/
│   │   ├── model_service.py             # LLM loading & streaming inference
│   │   ├── memory_service.py            # Session state management
│   │   ├── flight_api.py                # Flight search API client
│   │   ├── airport_search_service.py    # Airport lookup from CSV
│   │   └── safety_service.py            # Safety scoring service
│   ├── prompts/
│   │   └── system_prompts.py            # Travel agent system prompt
│   ├── components/
│   │   └── location_picker/             # Custom Streamlit Leaflet component
│   └── data/
│       ├── compiled_model_ready/        # City-level safety/demographic data
│       └── global_data/                 # Country-level macro indicators
├── safety/                              # Safety model training & evaluation
├── Makefile                             # Build automation
├── environment.yml                      # Conda dependencies
└── docker-compose.yml                   # Flight API scraper service
```

## Prerequisites

### Python (via Conda)
This project uses Python 3.12.8. Conda handles the version automatically.

If you are already using Anaconda or another conda distribution, skip to [Quick Setup](#quick-setup). Otherwise, install [Miniconda](https://docs.anaconda.com/miniconda/install/):

```bash
mkdir -p ~/miniconda3

# Apple Silicon (M1/M2/M3/M4)
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh

# Intel Mac
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh

bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

You know conda is installed and working if you see `(base)` in your terminal prompt.

### Make
Usually pre-installed on macOS/Linux. Check with `make -v`. If not installed:
```bash
brew install make
```

### Docker
Required for the flight search API.

1. Install [Docker Desktop](https://docs.docker.com/get-docker/)
2. Verify: `docker --version`

## Quick Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd wayfinder

# 2. Create the conda environment
make create
conda activate wayfinder

# 3. Start the flight API container
make docker-compose-up

# 4. Run the app
make run
```

On first launch, the Qwen 2.5 1.5B model (~3 GB) will be downloaded from HuggingFace automatically.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `localhost:8080` | Flight search API host |
| `API_BASE_SCHEME` | `http` | Flight API URL scheme |
| `WAYFINDER_DEVICE` | *(auto-detect)* | Force a compute device: `cuda`, `mps`, or `cpu` |
| `WAYFINDER_NO_MPS` | `false` | Set to `1` or `true` to skip MPS and fall back to CPU |

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make create` | Create the conda environment from `environment.yml` |
| `make update` | Update the conda environment from `environment.yml` |
| `make clean` | Remove the conda environment |
| `make activate` | Print the conda activate command (does **not** activate — you must run `conda activate wayfinder` yourself) |
| `make deactivate` | Deactivate the conda environment |
| `make run` | Start the Streamlit app |
| `make docker-compose-up` | Start the flight API Docker container |
| `make notebook` | Launch Jupyter Notebook |
| `make freeze` | Export installed packages to `environment.yml` |
| `make verify` | List conda environments to check the active one |

## Example Workflows

### First time setup
```bash
conda init --all
make create
conda activate wayfinder
make verify
python --version   # Should show 3.12.8
make docker-compose-up
make run
```

### Installing a new package
```bash
# Verify you're in the right environment
make verify

# Install via conda (preferred)
conda install <package_name>

# If you get a PackagesNotFoundError, use pip instead — conda will still
# track it in the environment properly
pip install <package_name>

# To remove a package
conda remove <package_name>
# or: pip uninstall <package_name>

# Export to environment.yml so teammates get it too
make freeze
```

### Daily development
```bash
# Before starting
git pull origin main
conda deactivate
make update
conda activate wayfinder
make docker-compose-up
make run

# After finishing
conda deactivate
make freeze   # only if you added/updated packages
git add .
git commit -m "your commit message"
git push origin <branch_name>
```

## Contributors

<table>
  <tr>
    <td>
        <a href="https://github.com/IanRebmann.png">
          <img src="https://github.com/IanRebmann.png" width="100" height="100" alt="Ian Rebmann"/><br />
          <sub><b>Ian Rebmann</b></sub>
        </a>
      </td>
     <td>
      <a href="https://github.com/omarsagoo.png">
        <img src="https://github.com/omarsagoo.png" width="100" height="100" alt="Omar Sagoo"/><br />
        <sub><b>Omar Sagoo</b></sub>
      </a>
    </td>
    <td>
      <a href="https://github.com/Ajmaljalal.png">
        <img src="https://github.com/Ajmaljalal.png" width="100" height="100" alt="Ajmal Jalal"/><br />
        <sub><b>Ajmal Jalal</b></sub>
      </a>
    </td>
  </tr>
</table>
