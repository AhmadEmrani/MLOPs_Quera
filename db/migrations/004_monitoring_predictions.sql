-- Explicit, additive migration. Apply only through approved DB migration path,
-- BEFORE deploying the modified batch scorer. No DROP or backfill is performed.
BEGIN;
CREATE TABLE IF NOT EXISTS monitoring_predictions (
    batch_id VARCHAR(64) NOT NULL,
    order_id VARCHAR(128) NOT NULL REFERENCES orders(order_id),
    model_version VARCHAR(128) NOT NULL,
    risk_probability DOUBLE PRECISION NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    scored_as_of TIMESTAMPTZ NOT NULL,
    feature_contract_sha256 CHAR(64) NOT NULL,
    feature_hash CHAR(64) NOT NULL,
    features_jsonb JSONB NOT NULL,
    persisted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (batch_id, order_id, model_version)
);
CREATE INDEX IF NOT EXISTS idx_monitoring_model_company_window
    ON monitoring_predictions (model_version, scored_as_of, order_id);
COMMIT;
