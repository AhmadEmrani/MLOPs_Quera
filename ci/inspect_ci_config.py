
"""Report GitLab CI variable names and presence, never values."""

import os

TARGETS = (
    "AIRFLOW_CONN_PROMISEOPS_FEED",
    "AIRFLOW_CONN_PROMISEOPS_POSTGRES",
    "AIRFLOW_CONN_PROMISEOPS_JOBS_KUBERNETES",
    "PROMISEOPS_FEED_URL",
    "PROMISEOPS_FEED_TOKEN",
    "PROMISEOPS_GATEWAY_URL",
    "PROMISEOPS_DATABASE_URL",
    "FEED_BASE_URL",
    "FEED_TOKEN",
    "GATEWAY_URL",
    "DATABASE_URL",
    "MLFLOW_TRACKING_URI",
)

KEY_HINTS = (
    "FEED",
    "GATEWAY",
    "POSTGRES",
    "PROMISEOPS",
    "MLFLOW",
)

print("=== GitLab CI variable presence ===")

for name in TARGETS:
    state = "SET" if os.environ.get(name) else "NOT SET"
    print(f"{name}: {state}")

print("\n=== Other relevant variable names ===")

found = sorted(
    name
    for name in os.environ
    if any(hint in name.upper() for hint in KEY_HINTS)
    and name not in TARGETS
)

print("names:", found if found else "none")

print(
    "\nNOTE: This checks GitLab CI, "
    "NOT the Airflow task environment."
)
