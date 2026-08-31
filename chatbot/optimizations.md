# What changed, and why

This replaces the earlier planning document. That file proposed changes; this one
records what was actually done.

Everything here is in place. Nothing needs a new service — no Redis, no queue.
Two database migrations are required (see **Before you deploy** at the bottom).

---

## Bugs fixed

These were real defects, not slowness.

**`/chat` crashed on payment receipts.** The endpoint was `async def` but called
the agent directly. Inside, `verify_and_save_payment_proof` called
`asyncio.run()`, which raises `RuntimeError: asyncio.run() cannot be called from
a running event loop` when a loop is already running. The WhatsApp path happened
to work because it ran in a worker thread, so the bug only appeared on one route.
Coroutines now go through `utils/async_bridge.py`, which runs them on a dedicated
loop and works from any thread.

**Duplicate orders.** WhatsApp redelivers a webhook it thinks failed. Nothing
checked whether a message had already been handled, so a redelivery re-ran the
customer's message — and could place a second order. Message ids are now
remembered for 15 minutes and repeats are dropped before any work happens.

**Long replies were never delivered.** WhatsApp rejects a text body over 4096
characters outright. A long product list or order history simply failed to send,
with no fallback. Replies are now split on paragraph, line, or sentence
boundaries and sent as several messages.

**The event loop was blocked on every request.** `/chat` called the blocking
agent inline, and the WhatsApp webhook downloaded the image and ran a vision
model call — several seconds — before returning 200. During that time no other
request in the process could be served, and the webhook risked WhatsApp's
timeout, which triggers the redelivery described above. All blocking work now
runs on worker threads.

**Sessions leaked.** Sessions lived in a module-level dict that only ever grew.
It recorded `last_activity` but nothing read it, so every phone number that ever
messaged kept a full agent object alive for the life of the process. Sessions now
have a sliding one-hour lease, a count cap, and a background sweeper.

**`/whatsapp/send-message` and `/whatsapp/profile/{n}` always failed.** Both
called service methods without the required `phone_number_id`, raising
`TypeError` on every call. They now accept it, defaulting to the single
configured account when there is only one.

**Unknown `AI_PROVIDER` failed at the worst moment.** If `AI_PROVIDER` was
neither `GPT` nor `DEEPSEEK`, the module-level `llm` name was never assigned. The
app imported cleanly and then died with `NameError` on the first customer
message. It now fails at startup with a message saying what to set.

**A database blip meant total silence.** The seller lookup raised, and the error
handler called that same lookup again — so the customer got no reply at all. The
error path no longer touches anything that can fail.

**Intent detection misread every order.** `'order'` appeared in three different
keyword lists and `order_tracking` was checked first, so "I want to order a
laptop" was classified as order tracking. Patterns are now ordered
most-specific-first and use word boundaries.

**RAG distances were meaningless.** Queries were encoded with
`normalize_embeddings=True` while the FAISS index held raw vectors, so the
distances compared against the threshold were between vectors of different
magnitudes. Stored vectors are now normalised to match.

**`@lru_cache` on a method leaked.** `FastVectorStore.similarity_search` was
decorated with `lru_cache`, which caches on `self` and therefore pinned the
instance and its embedding arrays forever — sitting on top of a second
hand-rolled dict cache doing the same job with different keys.

**Retrying a WhatsApp send could double-send.** Fixed as part of adding retries:
read timeouts are not retried (the message may already have been delivered), only
connect failures and statuses that mean "not processed" — 429, 502, 503, 504.

**Products were shown to customers as "Unknown Product".** The product template
read `name` and `id`, while `get_product_info` returns `product` and
`product_id`. Every product detail message said "Product ID: N/A / Name: Unknown
Product" with the right price and stock underneath it. Both spellings are
accepted now, and the new template layer renders from the structured data
directly.

**Entity extraction always came back empty for products.** The tool wrapper
stored the *image URL list* where the product id was expected, so
`chat_logs.entities` never recorded a product. Now `{"product_id": "3"}`.

**Model names were recorded doubled.** `langchain-openai` pointed at OpenRouter
reports `response_metadata["model_name"]` as the name twice over
(`openai/gpt-4o-miniopenai/gpt-4o-mini`), which would have become its own row in
every per-model cost report. Found while verifying against the live API;
normalised now. Costs were unaffected.

---

## Performance

**The agent was rebuilt for every message.** Each `/chat` request constructed
about twenty Pydantic model classes, twenty tools, a bound LLM and an executor,
then threw it all away. Building a Pydantic model class compiles a validator — it
is one of the more expensive things you can do per request, and the result was
identical every time.

- Argument schemas moved to `agent/schemas.py`, built once at import.
- The system prompt moved to `agent/prompts.py` and is cached per seller.
- Sessions are reused, so an agent is built once per conversation.

**The database had no connection pooling configured.** `create_engine` took no
pool arguments, so it ran on defaults: 5 connections, 10 overflow, and no
liveness check. `config/performance.py` defined a `DB_CONFIG` with sensible
values that nothing ever read. Now configured, with `pool_pre_ping` so a
connection dropped by Postgres or a NAT timeout reconnects instead of failing a
customer's message, and a connect timeout so an unreachable database fails fast
instead of pinning a worker thread for the OS TCP timeout.

**No indexes on anything the bot queries.** "This customer's pending orders" and
"this seller's products" were sequential scans. Product lookup uses
`name ILIKE '%term%'`, which cannot use a btree index at all — a trigram index
now covers it. See `database/migrations/002_performance_indexes.sql`.

**Every outbound call opened a new connection.** Each WhatsApp message paid for a
full TLS handshake to graph.facebook.com — roughly 100–300ms of pure overhead,
several times per conversation turn. Now a pooled, keep-alive session.

**Two connections per product lookup.** `get_product_info` called
`get_product_images`, which opened a second session while the first was still
checked out — two pool slots per lookup, and a deadlock risk once the pool
saturated. One session now.

**A thread per message.** Each finished turn did
`threading.Thread(target=...).start()` to write its log row: an OS thread created
and destroyed per message, unbounded under load, and killed mid-write at
shutdown. One shared bounded pool now, drained on shutdown.

**The whole webhook payload was logged at INFO with `indent=2`.** Megabytes of
logs and real CPU spent formatting JSON nobody read. Now at DEBUG and truncated.
The log file also had no rotation, so it grew without limit.

**Other:** gzip on responses (product lists compress well), a short cache on the
seller-by-WhatsApp-number lookup that ran on every message, a 30-second cache on
product listings invalidated whenever stock changes, and the embedding model is
no longer loaded at all when `RAG_ENABLED=false`.

---

## Security

- **Webhook signature verification** (`X-Hub-Signature-256`). The webhook is a
  public URL that creates orders and reads customer data; without this, anyone
  who finds it can drive the bot as any phone number. Off by default so it does
  not silently break an existing deployment — set `VERIFY_WEBHOOK_SIGNATURE=true`
  and `WHATSAPP_APP_SECRET`. Compared with `hmac.compare_digest`.
- **Rate limiting per customer.** Every message costs an LLM call, so a script
  hammering the webhook is a bill, not just load.
- **CORS.** `allow_origins=["*"]` with `allow_credentials=True` is refused by
  browsers anyway. Credentials are now only enabled when real origins are
  configured, and production warns if it is left open.
- **Template rendering is restricted.** Templates are edited by shop staff, and
  `{message.__class__.__mro__}` is valid `str.format` syntax that would leak
  internals. Only bare `{placeholder}` names are substituted — no attribute
  access, no indexing, no format specs.

---

## New features

- **Customisable message templates** — edit what the bot says without a deploy.
  See [docs/MESSAGE_TEMPLATES.md](docs/MESSAGE_TEMPLATES.md).
- **Token and cost tracking per session and seller**, plus model routing and a
  daily budget cap. See [docs/COST_AND_USAGE.md](docs/COST_AND_USAGE.md).
- **Human handover.** The agent can call `escalate_to_human` for refunds,
  complaints, or anything its tools cannot do. The bot then stops auto-replying
  to that customer so staff can take over the thread. `GET /whatsapp/handoffs`
  lists open ones; `DELETE /whatsapp/handoffs/{phone}` closes one.
- **Conversation history survives restarts.** History lived only in memory, so a
  deploy mid-order lost everything the customer had said and the bot started
  asking for their details again. Now persisted and reloaded per session.
- **Read receipts and a typing indicator**, sent before the turn starts rather
  than after the reply. A silent chat makes customers resend.
- **Real health and metrics.** `/health` returned `healthy` unconditionally — it
  stayed green with the database unreachable, so a load balancer kept sending
  traffic to an instance that could not answer. `/health?deep=true` now checks
  the database, and `/metrics` exposes latency percentiles, cache hit rates and
  pool occupancy.
- **Graceful shutdown.** Queued log and usage writes finish instead of being
  killed mid-write.

---

## Removed

Dead code that misled:

- `config/performance.py` — nothing read it. Its values now live in
  `config/settings.py`, which is actually used.
- `services/chat.py` — 94 lines, entirely commented out, a stale copy of
  `routes/chat.py`.
- `agent/multi_agent.py` — superseded; nothing imported it.
- `requirements_optimized.txt` — pinned LangChain 0.1.0 against an installed
  1.3.18. Installing it would have broken the app.
- `log_query_async` in `repositories/tools.py` — unused, and referenced
  `asyncio` after that import was removed, so calling it would have raised
  `NameError`.

---

## Before you deploy

**1. Apply the migrations.** All are safe to re-run.

```bash
cd chatbot
python scripts/apply_migrations.py            # uses DATABASE_URL from .env
python scripts/apply_migrations.py --dry-run  # see what it would do first
```

The runner exists because `psql` is usually not on PATH on a Windows dev machine,
and because `002` uses `CREATE INDEX CONCURRENTLY`, which Postgres refuses inside
a transaction block — sending a whole file through a driver as one string counts
as one. The runner splits each file into statements (handling dollar-quoted
`DO $$ ... $$` blocks) and runs them in autocommit, which is what `psql -f` does.

With `psql` available, this works too — but not with `-1` / `--single-transaction`:

```bash
psql "$DATABASE_URL" -f database/migrations/002_performance_indexes.sql
psql "$DATABASE_URL" -f database/migrations/003_conversation_messages.sql
psql "$DATABASE_URL" -f database/migrations/004_token_usage_and_templates.sql
```

Without `003`, history will not persist (set `PERSIST_CONVERSATIONS=false` to
silence the warnings). Without `004`, `/usage/history` returns 503 and templates
fall back to the built-in defaults — the bot still works.

**`005` is optional but worth reading.** `chat_logs.customer_id` has a foreign
key to `customers.id`, and a customer only gets a `customers` row once they hand
over their details at their first order. So every turn before that point fails
its analytics INSERT with a foreign-key violation — the bot replies normally
(the write is on a background thread) but the row is lost. That is most of the
funnel: every browse, every price question, every abandoned conversation. `005`
drops that one constraint. It is optional because Prisma may re-add the FK on the
Admin Portal's next `migrate`, so mirror it in `prisma/schema.prisma` if you
apply it.

**2. The database now lives on port 6432.** `database/docker-compose.yml`
publishes 6432 instead of 5432, and `DATABASE_URL` was updated to match. A native
PostgreSQL install on this machine already holds 5432 and wins the bind, so the
container's mapping looked correct in `docker ps` while every client silently
reached that other server - which has no `assistant` database. The container
still listens on 5432 internally; only the host port changed.

The same compose file now also applies migrations 001-004 automatically on a
first-time start, and mounts `init.docker.sql` rather than `init.sql`. `init.sql`
is a full `pg_dumpall` cluster dump beginning with `CREATE ROLE postgres` and
`CREATE DATABASE assistant`; both already exist by the time it runs, and because
the entrypoint uses `ON_ERROR_STOP=1` the container exits during init. That path
had never worked - verified by starting a throwaway container on an empty volume,
which failed on `role "postgres" already exists` before the fix and came up with
the full schema, 23 indexes and all migrations after it.

If anything else points at the old port - the Admin Portal backend's
`DATABASE_URL`, a saved connection in a database GUI - update it to 6432 too.

**3. Review the new settings.** `.env.example` documents all of them. Defaults
preserve existing behaviour: `COST_STRATEGY=fixed` uses `CHAT_MODEL` exactly as
before, and `VERIFY_WEBHOOK_SIGNATURE=false` keeps the webhook working until you
set the app secret.

**4. Keep one worker.** Sessions, caches, rate limits and webhook deduplication
are per-process. With more than one uvicorn worker, a customer's session and
their duplicate message can land on different processes, which would reinstate
the duplicate-order bug. The app is already concurrent within a process via its
worker threads. Moving that state to Redis is the change that would make
multi-worker safe — the cache layer in `utils/cache.py` is deliberately shaped
like the subset of operations Redis offers, so a backend can be dropped in
without touching call sites.

**5. Run the tests.**

```bash
pip install -r requirements-dev.txt
pytest
```
