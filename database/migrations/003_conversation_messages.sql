-- Conversation history, so context survives a restart.
--
-- Chat history lived only in the chatbot's in-memory session dict. A deploy or
-- crash in the middle of an order dropped everything the customer had said, and
-- the bot would start over asking for their name and address again.
--
-- Safe to re-run.
--
-- Run with:  psql "$DATABASE_URL" -f database/migrations/003_conversation_messages.sql

CREATE TABLE IF NOT EXISTS public.conversation_messages (
    id          BIGSERIAL PRIMARY KEY,
    seller_id   VARCHAR      NOT NULL,
    customer_id VARCHAR      NOT NULL,
    role        VARCHAR(16)  NOT NULL,
    content     TEXT         NOT NULL,
    created_at  TIMESTAMP    NOT NULL DEFAULT NOW(),
    CONSTRAINT conversation_messages_role_check
        CHECK (role IN ('user', 'assistant', 'system'))
);

-- Loading a session reads the newest N rows for one (seller, customer) pair.
CREATE INDEX IF NOT EXISTS idx_conv_seller_customer_created
    ON public.conversation_messages (seller_id, customer_id, created_at DESC);

-- Used by the retention cleanup below.
CREATE INDEX IF NOT EXISTS idx_conv_created_at
    ON public.conversation_messages (created_at);

COMMENT ON TABLE public.conversation_messages IS
    'Per-turn chat history used to rebuild an agent session after a restart.';

-- ---------------------------------------------------------------------
-- Retention: this table grows forever otherwise. Old history has no value
-- once a conversation is long finished, so trim anything older than 30 days.
-- Run from cron/pg_cron, or call it from the app on a schedule.
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.prune_conversation_messages(retain_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    removed INTEGER;
BEGIN
    DELETE FROM public.conversation_messages
     WHERE created_at < NOW() - (retain_days || ' days')::INTERVAL;
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$ LANGUAGE plpgsql;
