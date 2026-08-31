-- Token/cost accounting and customisable outbound message templates.
--
-- token_usage answers "what did this seller's bot cost this month", which
-- nothing could answer before: no token counts were recorded anywhere.
--
-- message_templates lets a seller edit the wording the bot sends without a code
-- change. A row with seller_id IS NULL is the default for everyone; a row with a
-- seller_id overrides that default for that seller.
--
-- Safe to re-run.
--
-- Run with:  psql "$DATABASE_URL" -f database/migrations/004_token_usage_and_templates.sql

-- ---------------------------------------------------------------------
-- token_usage : one row per conversation turn
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.token_usage (
    id                BIGSERIAL PRIMARY KEY,
    seller_id         VARCHAR       NOT NULL,
    customer_id       VARCHAR       NOT NULL,
    model             VARCHAR       NOT NULL,
    prompt_tokens     INTEGER       NOT NULL DEFAULT 0,
    completion_tokens INTEGER       NOT NULL DEFAULT 0,
    total_tokens      INTEGER       NOT NULL DEFAULT 0,
    llm_calls         INTEGER       NOT NULL DEFAULT 1,
    -- NUMERIC, not double precision: these get summed into invoices and
    -- floating point drift in money is not acceptable.
    cost_usd          NUMERIC(12,6) NOT NULL DEFAULT 0,
    created_at        TIMESTAMP     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_token_usage_seller_created
    ON public.token_usage (seller_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_token_usage_customer_created
    ON public.token_usage (customer_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_token_usage_model
    ON public.token_usage (model);

COMMENT ON TABLE public.token_usage IS
    'Per-turn LLM token consumption and cost, aggregated per seller for billing.';

-- Daily cost per seller and model - what a billing screen wants to read.
CREATE OR REPLACE VIEW public.token_usage_daily AS
SELECT
    seller_id,
    date_trunc('day', created_at)::date AS usage_date,
    model,
    COUNT(*)                  AS turns,
    SUM(llm_calls)            AS llm_calls,
    SUM(prompt_tokens)        AS prompt_tokens,
    SUM(completion_tokens)    AS completion_tokens,
    SUM(total_tokens)         AS total_tokens,
    SUM(cost_usd)             AS cost_usd
FROM public.token_usage
GROUP BY seller_id, date_trunc('day', created_at)::date, model;

COMMENT ON VIEW public.token_usage_daily IS
    'Daily token spend per seller and model.';

-- ---------------------------------------------------------------------
-- message_templates : seller-editable outbound copy
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.message_templates (
    id            SERIAL PRIMARY KEY,
    seller_id     VARCHAR      NULL,
    template_key  VARCHAR(64)  NOT NULL,
    body          TEXT         NOT NULL,
    enabled       BOOLEAN      NOT NULL DEFAULT TRUE,
    description   VARCHAR      NULL,
    updated_at    TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- A plain UNIQUE(seller_id, template_key) does not stop two rows with
-- seller_id IS NULL, because NULLs never equal each other. Two partial unique
-- indexes cover both cases.
CREATE UNIQUE INDEX IF NOT EXISTS uq_message_templates_seller_key
    ON public.message_templates (seller_id, template_key)
    WHERE seller_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_message_templates_default_key
    ON public.message_templates (template_key)
    WHERE seller_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_message_templates_key
    ON public.message_templates (template_key);

COMMENT ON TABLE public.message_templates IS
    'Customisable outbound message templates. seller_id NULL = global default.';
COMMENT ON COLUMN public.message_templates.body IS
    'Template text with {placeholder} fields. Unknown placeholders render empty.';
