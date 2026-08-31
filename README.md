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
├─ database/            # Postgres docker-compose, init SQL, migrations
└─ pgdb/                # Duplicate Postgres compose (prefer database/)
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
# Exposes Postgres on host port 6432 with DB=assistant, user=postgres, pass=12345678
# (6432 because a native PostgreSQL install often already holds 5432)
```

2) Backend API (Admin-Portal/backend)

```bash
cd Admin-Portal/backend
copy NUL .env  # Windows PowerShell: New-Item .env -ItemType File
```

Sample `.env`:
```env
# Database
DATABASE_URL=postgresql://postgres:12345678@localhost:6432/assistant
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
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell
pip install -r requirements.txt
copy .env.example .env        # then edit it
```

`chatbot/.env.example` documents every setting. The minimum is:

```env
# LLM (any OpenAI-compatible endpoint, e.g. OpenRouter)
AI_PROVIDER=GPT
API_KEY=your-api-key
API_BASE=https://openrouter.ai/api/v1
CHAT_MODEL=openai/gpt-4o-mini

DATABASE_URL=postgresql://postgres:12345678@localhost:6432/assistant
```

Apply the migrations (all safe to re-run):

```bash
python scripts/apply_migrations.py            # uses DATABASE_URL from .env
python scripts/apply_migrations.py --dry-run  # preview first
```

This does not need `psql` installed. If you do have it, note that `002` uses
`CREATE INDEX CONCURRENTLY`, which cannot run inside a transaction — so use
`psql -f` per file and never `-1` / `--single-transaction`.

`005` is **optional** and applied only if you ask for it by name
(`python scripts/apply_migrations.py 005`). It drops the foreign key from
`chat_logs.customer_id`, which currently makes every analytics row fail for a
customer who has not registered yet. Read the comments at the top of the file
first — Prisma may re-add the constraint on the Admin Portal's next migration.

Vector search (RAG) is optional and pulls in torch (~2.5GB). Only needed with
`RAG_ENABLED=true`:

```bash
pip install -r requirements-ml.txt
```

Run the service:
```bash
python start_server.py
# → Chatbot on http://localhost:8001
# Health check: http://localhost:8001/health?deep=true
```

Run the tests:
```bash
pip install -r requirements-dev.txt
pytest
```

## How the pieces connect

- Frontend calls Backend at `NEXT_PUBLIC_API_BASE_URL` (e.g., `http://localhost:5000/api`).
- Backend uses PostgreSQL via `DATABASE_URL` and optionally AWS S3 for uploads.
- Chatbot is an independent service exposing FastAPI endpoints (e.g., `/chat`). You can integrate the admin portal to call it directly (CORS is enabled in the chatbot) or via the backend.

## Important Paths and Ports

- Backend: `Admin-Portal/backend` → http://localhost:5000
- Frontend: `Admin-Portal/frontend` → http://localhost:3000
- Chatbot: `chatbot` → http://localhost:8001
- Database: `database/docker-compose.yml` → Postgres on host port 6432

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
  - `POST /chat` — chat over HTTP (`chatbot/routes/chat.py`)
  - `POST /whatsapp` — WhatsApp Cloud API webhook (alias: `/whatsapp/webhook`)
  - `GET /health?deep=true` — checks the database, LLM config and storage
  - `GET /metrics` — latency percentiles, cache hit rates, pool occupancy
  - `GET /templates` — customise what the bot says
  - `GET /usage/summary` — token usage and cost per seller
  - `GET /whatsapp/handoffs` — conversations waiting for a person
  - `GET /docs` — full interactive API reference
- Configure the provider in `.env` via `AI_PROVIDER`, `API_KEY`, `API_BASE`,
  `CHAT_MODEL`. See `chatbot/.env.example`.
- **Run one worker.** Sessions, caches, rate limits and WhatsApp webhook
  deduplication are per-process. The service is already concurrent within a
  process via its worker threads.

### Chatbot documentation

- [What changed and why](chatbot/optimizations.md) — bugs fixed, performance
  work, and the deployment steps
- [Message templates](chatbot/docs/MESSAGE_TEMPLATES.md) — edit the bot's
  wording without a deploy
- [Token usage and cost](chatbot/docs/COST_AND_USAGE.md) — per-seller cost
  reporting, model routing, budget caps

## Database

Two compose files exist (`database/` and `pgdb/`). **Use `database/`** - it is the
one kept current, and the only one that applies the migrations on a first-time
setup. They share a container name and cannot run at the same time.
```bash
cd database
docker compose up -d
```
On a **first-time** start (empty `pgdata` volume) the container runs, in order:
`init.docker.sql` (schema + seed data), then migrations 001-004.

It mounts `init.docker.sql`, not `init.sql`: the latter is a full `pg_dumpall`
cluster dump beginning with `CREATE ROLE postgres` and `CREATE DATABASE
assistant`. The entrypoint has already created both, so those statements fail and
- because it runs psql with `ON_ERROR_STOP=1` - the container exits during init.

On an **existing** volume the init scripts do not run at all. Apply migrations
there with `cd chatbot && python scripts/apply_migrations.py`.

The container publishes host port **6432** (it still listens on 5432 inside).
A native PostgreSQL install commonly already holds 5432 and wins the bind, which
leaves the mapping looking correct in `docker ps` while every client silently
reaches the wrong server - one with no `assistant` database.

## Docker (per service)

- Backend: `Admin-Portal/backend/Dockerfile`
- Frontend: `Admin-Portal/frontend/Dockerfile`

You can containerize each service separately; no root-level compose is provided for all services together.

## Environment Variables Summary

- Backend: `DATABASE_URL`, `PORT`, `AWS_REGION`, `AWS_ACCESSKEYID`, `AWS_SECRETACCESSKEY`
- Frontend: `NEXT_PUBLIC_API_BASE_URL`
- Chatbot: see `chatbot/.env.example`, which documents all of them. Required: `API_KEY`, `DATABASE_URL`, and a `WHATSAPP_CONFIG_0_*` block for WhatsApp.
- Database (docker): `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` (set in compose)

## Common Troubleshooting

- Connection refused to Postgres: ensure `docker compose ps` shows the DB healthy and `DATABASE_URL` matches.
- CORS/auth issues: verify cookies are sent (`credentials: include`) and frontend `NEXT_PUBLIC_API_BASE_URL` is correct.
- Chatbot cold start: first call may be slow if models are not preloaded; set `PRELOAD_MODELS=true`.

## License

All Rights Reserved - Viewing Only

This project is proprietary software. You may view the source code for informational 
purposes only. Copying, using, modifying, or distributing this software is prohibited.

See [LICENSE](LICENSE) file for details. 
