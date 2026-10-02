#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python}" -m yieldsat_knowledge.validate --library yieldsat_knowledge/assets/library.json --mirror spec/yieldsat_knowledge_assets/library.json
"${PYTHON:-python}" -m pytest -q -p no:cacheprovider yieldsat_knowledge/tests
