#!/usr/bin/env python3
"""Generate version-pinned golden inference fixtures for human code review.

Run manually, review the generated file, then commit it. Never run this as an
unattended CI update; a changed model requires an explicit updated baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from promiseops.model_pipeline.bundle import load_bundle
from promiseops.model_pipeline.candidates import positive_class_probability
from promiseops.model_pipeline.preprocessing import feature_matrix


def freeze(bundle: Path, fixture: Path, output: Path) -> None:
    model, release = load_bundle(bundle)
    golden=json.loads(fixture.read_text(encoding="utf-8"))
    features=feature_matrix(pd.DataFrame([case["features"] for case in golden["cases"]]))
    scores=positive_class_probability(model, features)
    payload={
        "schema_version":1,
        "model_version":release["model_version"],
        "model_artifact_sha256":release["artifact_sha256"],
        "feature_fixture_sha256": __import__("hashlib").sha256(fixture.read_bytes()).hexdigest(),
        "probability_tolerance":1e-9,
        "cases":[{"case_id": case["case_id"],"risk_probability":float(score)}
                  for case,score in zip(golden["cases"],scores,strict=True)],
    }
    if any(not math.isfinite(row["risk_probability"]) or not 0 <= row["risk_probability"] <= 1
           for row in payload["cases"]):
        raise ValueError("non-finite or out-of-range golden output")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an approved golden model output")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("Generated NEW, version-pinned golden fixture for review: " + str(output))


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--features",type=Path,default=Path("fixtures/golden_features_v1.json"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    freeze(a.bundle,a.features,a.output)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
