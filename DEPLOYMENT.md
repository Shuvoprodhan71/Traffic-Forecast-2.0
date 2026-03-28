# TrafficAI — Deployment Guide

Traffic Speed Forecasting Dashboard using RF + LSTM models on the METR-LA dataset.  
All inference runs **entirely in the browser** via TensorFlow.js — no GPU server required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + Vite + Tailwind CSS 4 + shadcn/ui |
| Backend | Node.js + Express + tRPC 11 |
| Database | MySQL / TiDB (via Drizzle ORM) |
| Auth | OAuth (optional — can be removed) |
| ML Inference | TensorFlow.js (browser) + JSON Random Forest (browser) |

---

## Prerequisites

- **Node.js** ≥ 18
- **pnpm** ≥ 9 (`npm install -g pnpm`)
- A **MySQL** database (local or cloud — PlanetScale, TiDB Cloud, Railway MySQL, etc.)

---

## Local Development Setup

### 1. Install dependencies

```bash
pnpm install
```

### 2. Configure environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Required variables:

```env
# Database
DATABASE_URL=mysql://user:password@host:3306/dbname

# Auth (JWT secret — generate any random string)
JWT_SECRET=your-random-secret-here

# OAuth (set to any placeholder if not using Manus OAuth)
VITE_APP_ID=your-app-id
OAUTH_SERVER_URL=https://api.manus.im
VITE_OAUTH_PORTAL_URL=https://manus.im
OWNER_OPEN_ID=your-open-id
OWNER_NAME=your-name

# Built-in API (leave blank if not using Manus platform)
BUILT_IN_FORGE_API_URL=
BUILT_IN_FORGE_API_KEY=
VITE_FRONTEND_FORGE_API_KEY=
VITE_FRONTEND_FORGE_API_URL=
```

### 3. Push database schema

```bash
pnpm db:push
```

### 4. Start the dev server

```bash
pnpm dev
```

Visit `http://localhost:3000`

---

## Production Build

```bash
pnpm build
```

This produces:
- `dist/` — compiled frontend (static files)
- `dist/index.js` — compiled backend server (ESM)

### Start production server

```bash
node dist/index.js
```

Set `NODE_ENV=production` and ensure all environment variables are set.

---

## Deployment Options

### Option A — Railway (recommended, easiest)

1. Push your code to GitHub
2. Create a new project on [railway.app](https://railway.app)
3. Connect your GitHub repo
4. Add a MySQL plugin (Railway provides one)
5. Set all environment variables in Railway's dashboard
6. Railway auto-detects `pnpm build` and `node dist/index.js`
7. Deploy — done

### Option B — Render

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Build command: `pnpm install && pnpm build`
4. Start command: `node dist/index.js`
5. Add a PostgreSQL or MySQL database add-on
6. Set environment variables in Render dashboard

### Option C — Vercel (frontend only)

> Note: Vercel is serverless — the Express backend will not run as-is.  
> Use this only if you remove the backend and make the app fully static.

For a **full-stack** deployment, use Railway or Render instead.

### Option D — VPS / Docker

Build the Docker image:

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install -g pnpm && pnpm install && pnpm build
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

```bash
docker build -t trafficai .
docker run -p 3000:3000 --env-file .env trafficai
```

---

## ML Models

The trained models are hosted on CDN and loaded automatically in the browser:

| File | Description | Size |
|---|---|---|
| `scaler_params.json` | MinMaxScaler parameters | ~2 KB |
| `rf_model.json` | Random Forest (50 trees, 72 features) | ~1.8 MB |
| `lstm_model_v2.json` | LSTM topology (Keras 2 format for TF.js) | ~12 KB |
| `group1-shard1of1.bin` | LSTM weights binary | ~115 KB |

All model files are served from CloudFront CDN. No server-side inference is required.

---

## Running Tests

```bash
pnpm test
```

35 unit tests covering: scaler round-trip, RF tree traversal, cyclical time encoding, CSV parsing, batch inference logic.

---

## Project Structure

```
client/src/pages/Home.tsx   ← Main dashboard (all 4 tabs: Predict, History, Batch, About)
server/routers.ts           ← tRPC API procedures
drizzle/schema.ts           ← Database schema
server/db.ts                ← Database query helpers
```

---

## Notes

- The **"Preview mode"** banner you see in the Manus editor is injected by the Manus hosting environment and will **not appear** on your own deployment.
- The dashboard works without a database — the DB is only used for user auth sessions. If you remove auth, you can simplify the backend significantly.
- All traffic forecasting inference is 100% client-side (browser). The backend is only needed for auth and session management.
