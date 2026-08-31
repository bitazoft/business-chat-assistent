-- Indexes for the queries the chatbot runs on every message.
--
-- Before this, "find this customer's pending orders" and "find this seller's
-- products" were sequential scans, and product name lookup (ILIKE '%term%')
-- could not use an index at all.
--
-- Safe to re-run: every statement uses IF NOT EXISTS.
--
-- Run with:  psql "$DATABASE_URL" -f database/migrations/002_performance_indexes.sql
--
-- CONCURRENTLY builds without locking out writes, but cannot run inside a
-- transaction block. Do NOT run this file with `psql -1` / --single-transaction.

-- ---------------------------------------------------------------------
-- orders
-- ---------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_customer_id
    ON public.orders (customer_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_seller_id
    ON public.orders (seller_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_status
    ON public.orders (status);

-- The hottest order query: this customer's orders in this status.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_customer_status
    ON public.orders (customer_id, status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_seller_created
    ON public.orders (seller_id, created_at DESC);

-- ---------------------------------------------------------------------
-- order_items
-- ---------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_order_items_order_id
    ON public.order_items (order_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_order_items_product_id
    ON public.order_items (product_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_order_items_order_product
    ON public.order_items (order_id, product_id);

-- ---------------------------------------------------------------------
-- products
-- ---------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_products_seller_id
    ON public.products (seller_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_products_seller_name
    ON public.products (seller_id, name);

-- ---------------------------------------------------------------------
-- item_img  (product images, fetched with every product lookup)
-- ---------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_img_product_id
    ON public.item_img (product_id);

-- ---------------------------------------------------------------------
-- seller_profiles  (looked up on every inbound WhatsApp message)
-- ---------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seller_profiles_whatsapp_number_id
    ON public.seller_profiles (whatsapp_number_id);

-- ---------------------------------------------------------------------
-- chat_logs  (written on every turn, read by the Admin Portal analytics)
-- ---------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_logs_customer_id
    ON public.chat_logs (customer_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_logs_seller_timestamp
    ON public.chat_logs (seller_id, timestamp DESC);

-- ---------------------------------------------------------------------
-- Trigram index so product search stops scanning the whole table.
--
-- Product lookup uses `name ILIKE '%term%'`. A leading wildcard cannot use a
-- btree index; a GIN trigram index can. Needs the pg_trgm extension, which
-- requires elevated rights - wrapped so a permission error leaves the rest of
-- this migration applied rather than aborting it.
-- ---------------------------------------------------------------------
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'Skipping pg_trgm: needs a superuser. Product search will still work, just without the trigram index.';
END
$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        CREATE INDEX IF NOT EXISTS idx_products_name_trgm
            ON public.products USING gin (name gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS idx_products_description_trgm
            ON public.products USING gin (description gin_trgm_ops);
    END IF;
END
$$;

-- Refresh planner statistics so the new indexes get used immediately.
ANALYZE public.orders;
ANALYZE public.order_items;
ANALYZE public.products;
ANALYZE public.chat_logs;
