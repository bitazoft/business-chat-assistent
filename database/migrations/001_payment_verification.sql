-- Payment proof verification fields.
-- Adds the columns the vision-based receipt check writes to.
-- Safe to re-run: every statement uses IF NOT EXISTS.

-- NOTE: payment_status already exists and is owned by the Admin Portal
-- (Paid / Pending / Processing / Failed / Rejected). We do not redefine it here;
-- the verification result lives in its own payment_verification column, and the
-- verifier sets payment_status to Paid or Pending to match the portal's vocabulary.

ALTER TABLE public.orders
    ADD COLUMN IF NOT EXISTS payment_verification  VARCHAR,
    ADD COLUMN IF NOT EXISTS payment_amount        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS payment_currency      VARCHAR,
    ADD COLUMN IF NOT EXISTS payment_reference     VARCHAR,
    ADD COLUMN IF NOT EXISTS payment_bank          VARCHAR,
    ADD COLUMN IF NOT EXISTS payment_date          VARCHAR,
    ADD COLUMN IF NOT EXISTS payment_flagged       BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS payment_flag_reason   VARCHAR,
    ADD COLUMN IF NOT EXISTS payment_verified_at   TIMESTAMP,
    ADD COLUMN IF NOT EXISTS payment_raw_extraction JSONB;

COMMENT ON COLUMN public.orders.payment_verification IS
    'Vision check result: verified | amount_mismatch | not_a_receipt | unreadable';
COMMENT ON COLUMN public.orders.payment_flagged IS
    'TRUE when the receipt needs a human to review it';
COMMENT ON COLUMN public.orders.payment_raw_extraction IS
    'Full JSON returned by the vision model, kept for auditing';

-- Lets staff pull up the review queue quickly
CREATE INDEX IF NOT EXISTS idx_orders_payment_flagged
    ON public.orders (payment_flagged)
    WHERE payment_flagged = TRUE;
