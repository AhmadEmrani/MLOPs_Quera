-- Run with an approved schema-migration identity before deploying new app code.
-- NULL means legacy per-order batch provenance was never captured; do NOT
-- backfill a guessed receipt timestamp for every individual order.
ALTER TABLE outcomes ADD COLUMN IF NOT EXISTS batch_available_at TIMESTAMPTZ;
-- New rows record the envelope-level batch_available_at transactionally.
-- For legacy rows, historical queries use the latest recorded outcome receipt
-- as a conservative upper bound. It can delay/omit labels, never expose early.
