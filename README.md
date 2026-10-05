# ANTROUTE

ANTROUTE is an automated traffic data ingestion, preprocessing, and routing intelligence system targeting Metro Manila traffic conditions.

## 📂 Repository Structure

Our repository follows a reproducible layout suitable for standard machine learning and data engineering projects:

```
ANTROUTE/
├── .github/
│   └── pull_request_template.md    # Standard PR template
├── pipeline_context.md             # NLP data preprocessing context and schema logs
├── traffic_system/
│   ├── configs/                    # Configuration files (YAML, JSON, env templates)
│   ├── data/
│   │   ├── raw/                    # Immutable raw data (CCTV footages, raw tweets/news)
│   │   └── processed/              # Cleaned, normalized, and tokenized datasets
│   ├── docs/                       # Project documentation, architecture diagrams, and meeting notes
│   ├── experiments/                # Scripts for running model training experiments and hyperparameter tuning
│   ├── notebooks/                  # Jupyter notebooks for EDA and rapid prototyping
│   ├── src/                        # Core source code
│   │   ├── ingestion/              # Data scrapers (Twitter, GDELT, CCTV APIs)
│   │   ├── preprocessing/          # Text and Vision cleaners, tokenizers, normalizers
│   │   ├── models/                 # Model architectures (CNN, LSTM, ViT)
│   │   ├── routing/                # Traffic routing algorithms and graph heuristics
│   │   └── vision/                 # Visual data pipeline (Frame extractors, Patch Embedders)
│   ├── tests/                      # Unit and integration tests
│   ├── requirements.txt            # Python dependencies
│   └── pipeline_runner.py          # Unified runner for data pipelines
└── README.md                       # This file
```

---

## 🌿 Branching Strategy & Workflow

To maintain code quality and stability, we use a strict **Feature Branch Workflow** and require PR reviews before merging into the main branch.

### 1. Main Branch Protection
- The `main` branch is **protected**. Direct pushes to `main` are disabled.
- All code changes must be integrated via a Pull Request (PR).
- PRs require at least **one approving review** before merging.
- Continuous Integration (CI) status checks (e.g., tests) must pass before a merge is allowed.

### 2. Issue and Ticket ID Requirement
All work must trace back to a tracking issue or Jira ticket.
- **Branch Naming**: Prefix your branch name with the ticket ID.
  * Format: `<ticket-id>/<short-description>` or `<type>/<ticket-id>-<short-description>`
  * Example: `ANT-123/add-cctv-extractor` or `feat/ANT-456-patch-embedder`
- **PR Title**: Must include the ticket ID (e.g., `[ANT-123] Implement CNN+LSTM DataLoader`).

### 3. Workflow Steps
Follow this pipeline for any change:

1. **Feature Branch**: Create a new branch from `dev` matching the naming convention.
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/ANT-001-setup-repo
   ```
2. **Develop & Commit**: Make your changes. Write unit tests for new features. Commit with descriptive messages.
3. **Pull Request**: Push your branch to the remote repository and open a Pull Request against `dev`.
4. **Review**: Ensure the PR description satisfies the **Pull Request Template**. Request a review from a team member.
5. **Test**: Wait for CI checks/tests to pass. Address any reviewer feedback.
6. **Merge**: Once approved and tests pass, squash and merge into `dev`.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- `ffmpeg` (Required for video preprocessing in the vision pipeline)
- PyTorch & Transformers

### Installation
```bash
git clone <repo-url>
cd ANTROUTE/traffic_system
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Tech Stack

### Backend

- **Framework:** Python, FastAPI
- **Database:** PostgreSQL (with PostGIS/GeoAlchemy2 for spatial data), AsyncPG
- **Caching/Queue:** Redis
- **Server:** Uvicorn

### Frontend

- **Framework:** React Native (Expo)
- **Styling:** TailwindCSS
- **Language:** TypeScript

## Setup and Run Instructions

### Prerequisites

- Docker and Docker Compose
- Node.js and npm (for frontend local development)
- Python 3.12+ (for backend local development)

### Running with Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ANTROUTE
   ```
2. Start the supporting services using Docker Compose:

   ```bash
   docker-compose up -d
   ```

   This starts PostgreSQL and Redis. The backend and frontend are run locally during development.

3. Start the backend in a separate terminal:

   ```bash

   ```

4. Start the frontend in another terminal:
   ```bash
   cd frontend
   npm install
   npx expo start
   ```

### New Contributor Run Order

1. Start Docker with `docker-compose up -d`.
2. Start the backend server with Uvicorn.
3. Start the Expo frontend with `npm start`.

### Existing Developer Run Flow

If you already have the project cloned and dependencies installed, use this shorter flow:

1. Start the supporting services:
   ```bash
   docker-compose up -d
   ```
2. Start the backend from the `backend` directory:

   ```bash

   ```

3. Start the frontend from the `frontend` directory:
   ```bash
   npm start
   ```
