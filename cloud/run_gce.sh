#!/bin/bash
# QTbtx Run Script for GCE
# Usage: ./run_gce.sh [signal_type] [data_bucket]

SIGNAL=${1:-"all"}
BUCKET=${2:-"qtbtx-data"}

echo "============================================="
echo "QTbtx Optimizer - Google Cloud"
echo "============================================="
echo "Signal: $SIGNAL"
echo "Bucket: gs://$BUCKET"
echo "Started: $(date)"
echo ""

cd ~/QTbtx

# Activate environment
source venv/bin/activate

# Download latest data from GCS
echo "Downloading data from Cloud Storage..."
gsutil -m cp -r gs://$BUCKET/preprocessed/* data/preprocessed/

# Run optimization
echo ""
echo "Running optimization..."
python run_optimizer.py --signal $SIGNAL --data data/preprocessed --output results

# Upload results to GCS
echo ""
echo "Uploading results to Cloud Storage..."
gsutil -m cp results/*.xlsx gs://$BUCKET/results/

echo ""
echo "============================================="
echo "Completed: $(date)"
echo "Results at: gs://$BUCKET/results/"
echo "============================================="

