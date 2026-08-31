# Token usage and cost

Nothing measured token use before, so there was no way to tell what a
conversation cost or which seller was driving the bill. Now every turn is
recorded, and there are four mechanisms for bringing the cost down.

## What gets recorded

One row per **conversation turn**, in the `token_usage` table. A turn is one
customer message and the bot's reply — which may involve several model calls,
one per tool round trip. All of them are counted together, because the seller is
billed for the turn, not the internals.

Each row has: seller, customer, model, prompt tokens, completion tokens, number
of model calls, and cost in USD.

Live totals are also kept in memory, per session and per seller, for instant
reads. Those reset on restart; the table is the durable record.

## Reading it

```bash
# Live totals for every seller
curl http://localhost:8001/usage/summary

# One seller, including today's spend against their budget
curl http://localhost:8001/usage/seller/7

# What one customer's conversation has cost
curl http://localhost:8001/usage/session/7/94771234567

# The most expensive live conversations - good for spotting a loop
curl http://localhost:8001/usage/sessions/top

# Durable history from the table
curl "http://localhost:8001/usage/history?seller_id=7&days=30&group_by=day"
curl "http://localhost:8001/usage/history?days=7&group_by=model"
```

`group_by` accepts `day`, `model`, `customer`, `seller`.

There is also a SQL view for billing screens:

```sql
SELECT * FROM token_usage_daily WHERE seller_id = '7' ORDER BY usage_date DESC;
```

## Prices

Costs are computed from a per-model price table (USD per 1M tokens). See what is
configured:

```bash
curl http://localhost:8001/usage/models
```

Model names are matched loosely, so `openai/gpt-4o-mini` and
`gpt-4o-mini-2024-07-18` both resolve to `gpt-4o-mini`.

**Check the built-in prices against your provider's current price list.** They
are a starting point, not a live feed. To override or add models, point
`MODEL_PRICING_FILE` at a JSON file:

```json
{
  "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
  "some-provider/some-model": {"input": 1.0, "output": 3.0}
}
```

An unknown model is billed at 0 and logs a warning — better an obvious zero than
a confidently wrong number in an invoice.

---

## Reducing the cost

### 1. Shortcut replies — free

"hi", "thanks", "ok" are a real share of WhatsApp traffic and need no model at
all. They are answered from a template for zero tokens.

Only messages that are *nothing but* a greeting or thanks take this path. "hi, do
you have Ceylon tea?" goes to the agent. Greetings only shortcut at the start of
a conversation — a mid-conversation "hi" usually means the customer is chasing a
reply, and the agent should see that.

```env
SHORTCUT_REPLIES_ENABLED=true   # default
```

### 2. Response cache — free on repeats

The same opening question ("do you deliver?") from different customers has the
same answer. Cached for 15 minutes.

**Only cached when the turn used no tools and had no history.** Any tool call
means the answer depended on that customer's orders, profile, or stock at that
moment, and must never be replayed to someone else. The cache is also scoped per
seller.

```env
RESPONSE_CACHE_ENABLED=true
RESPONSE_CACHE_TTL=900
```

### 3. Model routing

Set with `COST_STRATEGY`:

**`fixed`** (default) — always `CHAT_MODEL`. Same behaviour as before.

**`tiered`** — a cheap model for simple turns, a stronger one where mistakes cost
real money:

```env
COST_STRATEGY=tiered
MODEL_CHEAP=openai/gpt-4o-mini
MODEL_STRONG=openai/gpt-4o
```

It escalates when the turn is about placing an order, contains an image, or
mentions a refund, complaint, cancellation, receipt or payment — and when the
conversation has run past a dozen messages, which usually means it has either
gone wrong or is mid-order.

**`rotation`** — round-robin across a list, to spread spend and provider rate
limits:

```env
COST_STRATEGY=rotation
MODEL_ROTATION=openai/gpt-4o-mini,google/gemini-2.0-flash,meta-llama/llama-3.1-8b-instruct
```

Each model gets its own executor, built once per session and reused, so switching
between turns costs nothing.

### 4. Daily budget cap

```env
DAILY_BUDGET_USD=5.00
```

Per seller, per day. Past the cap, everything drops to the cheapest configured
model — the bot degrades rather than billing without limit. `0` disables it.

Check where a seller stands:

```bash
curl http://localhost:8001/usage/seller/7
```

### What's switched on

```bash
curl http://localhost:8001/usage/optimization
```

Shows the strategy, the models in play, the budget, and the response cache's hit
rate.

---

## Other things that affect the bill

**Rate limiting** is a spend control as well as a load control — one script
looping on the webhook is a real cost. Default is 20 messages per minute per
customer (`RATE_LIMIT_MESSAGES`, `RATE_LIMIT_WINDOW_SECONDS`).

**`MAX_CHAT_HISTORY`** (default 20) caps how many past messages go into each
prompt. History is resent on every turn, so this is a direct multiplier on
prompt tokens.

**`AGENT_MAX_ITERATIONS`** (default 8) caps tool round trips per turn. Each one
is another model call with the full prompt, so a runaway turn is expensive. The
counter is at `agent.iteration_limit` in `/metrics` — if it climbs, the agent is
getting stuck and every one of those turns is costing a full allowance.

**The system prompt is about 4KB and is sent on every turn.** It is the single
largest fixed cost per message. Trimming it is the biggest remaining saving
available, but it changes behaviour, so it is left alone here — measure before
and after if you try it.

**`LLM_MAX_TOKENS`** (default 512) caps the reply length, which caps completion
tokens — the more expensive half.

---

## Turning it off

```env
TRACK_TOKEN_USAGE=false
```

Live totals are still collected (they are free); this stops the database writes.
Usage recording never blocks a reply — it runs on the background executor, and a
failure there is logged, not surfaced to the customer.
