# ANTROUTE

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
