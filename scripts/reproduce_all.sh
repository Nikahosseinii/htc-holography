#!/usr/bin/env bash
set -euo pipefail

echo "=== HTC Holography paper reproducibility run ==="
echo "Date: $(date)"
echo "Python: $(python --version)"
echo "Working directory: $(pwd)"

mkdir -p logs figures tables reference_outputs

echo ""
echo "Running Fig. 3..."
bash scripts/reproduce_fig3_ils.sh 2>&1 | tee logs/fig3_ils.log

echo ""
echo "Running Fig. 4..."
bash scripts/reproduce_fig4_payload.sh 2>&1 | tee logs/fig4_payload.log

echo ""
echo "Running Fig. 5..."
bash scripts/reproduce_fig5_latency.sh 2>&1 | tee logs/fig5_latency.log

echo ""
echo "Running Hybrid AIF ILS figure..."
bash scripts/reproduce_fig8_hybrid_ils.sh 2>&1 | tee logs/fig8_hybrid_ils.log

echo ""
echo "Running tables..."
bash scripts/reproduce_tables.sh 2>&1 | tee logs/tables.log

echo ""
echo "Creating checksum file..."
sha256sum figures/* tables/* 2>/dev/null > reference_outputs/checksums.txt || true

echo ""
echo "Done. Check figures/, tables/, logs/, and reference_outputs/checksums.txt"
