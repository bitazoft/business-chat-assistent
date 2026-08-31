-- OPTIONAL. Keep analytics for customers who have not registered yet.
--
-- The problem
-- -----------
-- chat_logs.customer_id has a foreign key to customers.id. A customer only gets
-- a customers row when they hand over their name, email, address and phone -
-- which happens at their first order, if it happens at all.
--
-- So for every conversation before that point, each turn's INSERT into chat_logs
-- fails with:
--
--     insert or update on table "chat_logs" violates foreign key constraint
--     "chat_logs_customer_id_fkey"
--
-- The bot still replies normally (the write happens on a background thread and
-- the failure is logged, not surfaced), but the analytics row is lost. That is
-- most of the funnel: every browse, every price question, every abandoned
-- conversation.
--
-- The fix
-- -------
-- chat_logs is an append-only analytics log. It does not need referential
-- integrity with customers, and enforcing it is what loses the data. This drops
-- that one constraint and leaves the column and its index in place, so existing
-- queries and joins keep working.
--
-- Why this is optional
-- --------------------
-- It relaxes a constraint on an existing table, so it is your call. If the Admin
-- Portal relies on that FK (Prisma may re-add it on the next `prisma migrate`),
-- either skip this or mirror the change in prisma/schema.prisma.
--
-- Safe to re-run.
--
-- Run with:
--     cd chatbot && python scripts/apply_migrations.py 005

ALTER TABLE public.chat_logs
    DROP CONSTRAINT IF EXISTS chat_logs_customer_id_fkey;

-- The column and index stay: only the constraint is gone.
CREATE INDEX IF NOT EXISTS idx_chat_logs_customer_id
    ON public.chat_logs (customer_id);

COMMENT ON COLUMN public.chat_logs.customer_id IS
    'Customer identifier (WhatsApp number or web user id). Deliberately not a '
    'foreign key: conversations are logged before a customer ever registers.';
