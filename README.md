# Business Chat Assistant Monorepo

A full-stack monorepo for a business admin portal and AI-powered chatbot with RAG capabilities. It includes:
- Admin Portal (Next.js frontend + Node/Express backend with Prisma/PostgreSQL)
- Chatbot service (FastAPI + LangChain)
- Local PostgreSQL via Docker Compose

## Repository Structure

```
business-chat-assistent/
├─ Admin-Portal/
│  ├─ backend/          # Node.js + Express + Prisma API
│  └─ frontend/         # Next.js 15 app (TypeScript + Tailwind)
├─ chatbot/             # FastAPI + LangChain chatbot service
├─ database/            # Postgres docker-compose + init.sql
└─ pgdb/                # Alternative Postgres compose (same as database/)
```

## Tech Stack

- Frontend: Next.js 15, React 18, TypeScript, Tailwind, Radix UI
- Backend: Node.js, Express 5, Prisma, PostgreSQL, JWT auth, AWS S3
- Chatbot: Python, FastAPI, LangChain, sentence-transformers, FAISS, Uvicorn
- Database: PostgreSQL 17 (Docker)

## Prerequisites

- Node.js 18+
- Python 3.10+
- Docker + Docker Compose
- npm (or pnpm/yarn), pip/venv

## Quick Start (Local Dev)

1) Start PostgreSQL

```bash
cd database
docker compose up -d
# Exposes Postgres on 5432 with DB=assistant, user=postgres, pass=12345678
```

2) Backend API (Admin-Portal/backend)

```bash
cd Admin-Portal/backend
copy NUL .env  # Windows PowerShell: New-Item .env -ItemType File
```

Sample `.env`:
```env
# Database
DATABASE_URL=postgresql://postgres:12345678@localhost:5432/assistant
# Server
PORT=5000
# AWS S3 (optional for uploads)
AWS_REGION=your-region
AWS_ACCESSKEYID=your-access-key
AWS_SECRETACCESSKEY=your-secret-key
```

Install and run with Prisma:
```bash
npm install
npx prisma generate
npx prisma migrate dev --name init
# Optional seed (if applicable):
# npm run prisma:seed  # prisma.seed is configured to prisma/seed.cjs
npm run dev
# → Backend on http://localhost:5000
```

3) Frontend (Admin-Portal/frontend)

```bash
cd Admin-Portal/frontend
copy NUL .env.local
```

Sample `.env.local`:
```env
# Point to the backend API above
NEXT_PUBLIC_API_BASE_URL=http://localhost:5000/api
```

Install and run:
```bash
npm install
npm run dev
# → Frontend on http://localhost:3000
```

4) Chatbot service (chatbot)

```bash
cd chatbot
copy NUL .env
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell
pip install -r requirements.txt
```

Sample `.env`:
```env
# Server
HOST=0.0.0.0
PORT=8001
ENVIRONMENT=development
PRELOAD_MODELS=true
LOG_LEVEL=INFO
DEBUG_MODE=false

# LLM providers (set at least one)
OPENAI_API_KEY=your-openai-key
# or
DEEPSEEK_API_KEY=your-deepseek-key

# Optional: HuggingFace cache
HF_HOME=cache/huggingface
TOKENIZERS_PARALLELISM=false
```

Run the service:
```bash
python start_server.py
# → Chatbot on http://localhost:8001
# Health check: http://localhost:8001/health
```

## How the pieces connect

- Frontend calls Backend at `NEXT_PUBLIC_API_BASE_URL` (e.g., `http://localhost:5000/api`).
- Backend uses PostgreSQL via `DATABASE_URL` and optionally AWS S3 for uploads.
- Chatbot is an independent service exposing FastAPI endpoints (e.g., `/chat`). You can integrate the admin portal to call it directly (CORS is enabled in the chatbot) or via the backend.

## Important Paths and Ports

- Backend: `Admin-Portal/backend` → http://localhost:5000
- Frontend: `Admin-Portal/frontend` → http://localhost:3000
- Chatbot: `chatbot` → http://localhost:8001
- Database: `database/docker-compose.yml` → Postgres on 5432

## Backend Notes

- Entry: `index.js`; Routes under `routes/` mounted at `/api/*`
- CORS: allows `http://localhost:3000`
- Prisma: models in `prisma/schema.prisma`
- Env vars used:
  - `DATABASE_URL` (required)
  - `PORT` (default 5000)
  - `AWS_REGION`, `AWS_ACCESSKEYID`, `AWS_SECRETACCESSKEY` (for S3 uploads)

Common scripts:
```bash
npm run dev             # start dev server
npx prisma generate
npx prisma migrate dev
```

## Frontend Notes

- Next.js 15 with middleware protecting `/dashboard/*`
- Set `NEXT_PUBLIC_API_BASE_URL` to the backend API URL
- Scripts:
```bash
npm run dev
npm run build
npm start
```

## Chatbot Notes

- ASGI app: `main:app`
- Start options:
```bash
python start_server.py
# or
uvicorn main:app --reload --port 8001 --host 0.0.0.0
```
- Key endpoints:
  - `POST /chat` (see `chatbot/routes/chat.py`)
  - `GET /health`
- Configure provider keys in `.env` (OpenAI/DeepSeek)

## Database

Two equivalent compose files exist (`database/` and `pgdb/`). Prefer `database/`.
```bash
cd database
docker compose up -d
```
Initial SQL is mounted from `database/init.sql`.

## Docker (per service)

- Backend: `Admin-Portal/backend/Dockerfile`
- Frontend: `Admin-Portal/frontend/Dockerfile`

You can containerize each service separately; no root-level compose is provided for all services together.

## Environment Variables Summary

- Backend: `DATABASE_URL`, `PORT`, `AWS_REGION`, `AWS_ACCESSKEYID`, `AWS_SECRETACCESSKEY`
- Frontend: `NEXT_PUBLIC_API_BASE_URL`
- Chatbot: `HOST`, `PORT`, `ENVIRONMENT`, `PRELOAD_MODELS`, `LOG_LEVEL`, `DEBUG_MODE`, `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`, optional `HF_HOME`, `TOKENIZERS_PARALLELISM`
- Database (docker): `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` (set in compose)

## Common Troubleshooting

- Connection refused to Postgres: ensure `docker compose ps` shows the DB healthy and `DATABASE_URL` matches.
- CORS/auth issues: verify cookies are sent (`credentials: include`) and frontend `NEXT_PUBLIC_API_BASE_URL` is correct.
- Chatbot cold start: first call may be slow if models are not preloaded; set `PRELOAD_MODELS=true`.

## License

MIT Licence 
