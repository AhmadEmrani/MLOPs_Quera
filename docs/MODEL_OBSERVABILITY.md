# PromiseOps — model serving observability

Base: `refactor/ciConfigsTestAPI`. This change does not alter endpoints, API authentication, database schema, image release selection, or HPA. Use the same published model bundle and the usual code-only staging/production release path. No new dependencies.

## Metrics

| Metric | Meaning |
|---|---|
| `promiseops_feature_missing_total{feature="..."}` | Null values seen in each canonical input feature; all 15 children start at zero |
| `promiseops_feature_rows_total` | Rows inspected (including rows for a model computation that later fails) |
| `promiseops_model_inferences_total` | Successful fresh model computations, not unique orders or guaranteed durable inserts |
| `promiseops_model_inference_errors_total` | Computations raising or returning nonfinite risk |
| `promiseops_model_inference_seconds` | Histogram of model computation durations, including failed attempts; excludes DB and HTTP overhead |
| `promiseops_prediction_risk` | Histogram of successful fresh computations' served risk in [0,1] |
| `promiseops_model_loaded` | 1 if the model bundle is loaded in the process, else 0 |
| `promiseops_model_info{model_version="sha256:..."}` | Loaded immutable artifact digest; one trusted value per process; no per-request version labels |
| `promiseops_validation_reference_available` | 1 if the required recorded reference metrics are valid |
| `promiseops_validation_recall_at_10pct`, `promiseops_validation_precision_at_10pct` | Recorded validation-cohort ranking metrics |
| `promiseops_validation_average_precision`, `promiseops_validation_brier` | Recorded validation PR-AUC and Brier |
| `promiseops_validation_n`, `promiseops_validation_late_count`, `promiseops_validation_prevalence`, `promiseops_validation_k` | Recorded validation population, positives, prevalence and review-list size |

The version metadata is a bounded, release-controlled addition to the existing route/status/fixed-feature labels described by the contract. It is never taken from a request, cached prediction or arbitrary model name. If the course enforces a literal allowlist of only the three original label keys, confirm the model identity Info metric with staff; the immutable version also remains in response/log/evidence. Histograms use their standard `le` labels. No model identity is added to high-volume counters or histograms. `/metrics` still requires `X-API-Key`.

A cached request ID does not invoke `score()` again. Concurrent duplicate requests may compute more than once before the DB resolves the race: these counters deliberately measure computations, not unique orders. Never use them as a unique-order monitoring cohort. Missing reference fields are NaN (unknown), not zero. Existing legacy bundles without reference fields still serve; `reference_available` signals whether the reference is complete. Validation metrics are NOT live quality estimates.

## Grafana — Team Prometheus, job g02-production

Use `Instant` and Stat/Table for current metrics, Time series for rates and latencies. Prefer max across identical reference values only when the deployment is stable; use per-instance values to detect mixed versions in a rollout. Never sum replicated reference gauges.

```promql
# Loaded model identity: Table or Stat, legend {{model_version}}; each instance separately
promiseops_model_info{job="g02-production",model_version=~"sha256:[0-9a-f]{64}"}
# Loaded instances (not readiness)
sum(promiseops_model_loaded{job="g02-production"})
# Successful fresh computations per second
sum(rate(promiseops_model_inferences_total{job="g02-production"}[5m]))
# Missing observations by feature in the last hour (zero children exist on startup)
sum by (feature) (increase(promiseops_feature_missing_total{job="g02-production"}[1h]))
# Missing percentage per feature; deliberately no value when no rows were examined
100 * sum by (feature) (rate(promiseops_feature_missing_total{job="g02-production"}[5m]))
  / scalar(sum(rate(promiseops_feature_rows_total{job="g02-production"}[5m])) > 0)
# Model compute p95 in ms; no value when no fresh computations occurred
1000 * histogram_quantile(0.95, sum by (le) (rate(promiseops_model_inference_seconds_bucket{job="g02-production"}[5m])))
# Risk p50 and p95 (separate queries); not accuracy or calibrated live performance
histogram_quantile(0.50, sum by (le) (rate(promiseops_prediction_risk_bucket{job="g02-production"}[5m])))
histogram_quantile(0.95, sum by (le) (rate(promiseops_prediction_risk_bucket{job="g02-production"}[5m])))
# Model compute failures per second
sum(rate(promiseops_model_inference_errors_total{job="g02-production"}[5m]))
# Recorded validation reference; multiply by 100 only for percent display
promiseops_validation_reference_available{job="g02-production"}
promiseops_validation_recall_at_10pct{job="g02-production"}
promiseops_validation_precision_at_10pct{job="g02-production"}
promiseops_validation_average_precision{job="g02-production"}
promiseops_validation_brier{job="g02-production"}
promiseops_validation_n{job="g02-production"}
```

## Verification

1. On feature branch run `python -m pytest -q tests/test_model_observability.py tests/test_api.py` using the project venv, followed by the normal CI suite.
2. Deploy to Staging with the same approved model package (`RUN_MODEL_RELEASE=false`), verify existing API smoke tests. Production approval remains manual. The staff Prometheus job currently scrapes Production; use existing authenticated smoke or approved CI for Staging.
3. After Production rollout, check two `up` targets, two `model_loaded` gauges, their immutable SHA identity, and all feature children. Generate controlled prediction traffic with the existing acceptance harness if needed. A zero counter is not proof of good data without positive traffic; the missing ratio and risk histogram require traffic.
4. Save real Grafana/evidence, including model version, image digest, commit, timestamps, and load result. Do not include secret values.

## Separate remaining lifecycle task

The branch's `monitor.py` is a stub and `worker.py --mode monitor-train` is not implemented. `predictions.created_at` is wall time and scheduled scoring currently persists only the priority result, not every scored prediction with company replay time. Do not treat the full dataset's labels or a static validation manifest as live quality. Implement durable replay predictions with explicit company timestamps, as-of filtered outcomes, immutable reference feature distribution and independent seven-day windows, then persist and export report metrics. Required: minimum 100 unique orders/window, two nonoverlapping drift windows, PSI/TVD threshold 0.2, 60-day predictions and 100 unique mature labels with both classes, recorded retraining reasons, and existing candidate gates and human production approval. Those are not completed by this instrumentation change.
