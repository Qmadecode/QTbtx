# QTbtx Google Cloud Deployment Guide

## Option 1: Google Compute Engine (Recommended for long runs)

### Step 1: Create a VM Instance

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create VM with good specs for optimization (Frankfurt)
gcloud compute instances create qtbtx-optimizer \
    --zone=europe-west3-a \
    --machine-type=e2-standard-4 \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB
```

### Step 2: SSH into the VM

```bash
gcloud compute ssh qtbtx-optimizer --zone=europe-west3-a
```

### Step 3: Setup on VM

```bash
# Update and install Python
sudo apt update
sudo apt install -y python3 python3-pip git

# Clone the repo
git clone https://github.com/Qmadecode/QTbtx.git
cd QTbtx

# Install dependencies
pip3 install -r requirements.txt

# Create data directory
mkdir -p data/preprocessed
```

### Step 4: Upload Data

From your local machine:
```bash
# Upload data to VM
gcloud compute scp --recurse /path/to/your/preprocessed/*.csv qtbtx-optimizer:~/QTbtx/data/preprocessed/ --zone=europe-west3-a
```

### Step 5: Run Optimization

```bash
# Run all signals
python3 run_optimizer.py --signal all --data data/preprocessed --output results

# Or run specific signal
python3 run_optimizer.py --signal resist --data data/preprocessed --output results
```

### Step 6: Download Results

From your local machine:
```bash
gcloud compute scp --recurse qtbtx-optimizer:~/QTbtx/results/*.xlsx . --zone=europe-west3-a
```

---

## Option 2: Google Cloud Storage + Compute Engine

### Step 1: Upload Data to Cloud Storage

```bash
# Create bucket
gsutil mb gs://qtbtx-data

# Upload preprocessed data
gsutil -m cp -r /path/to/preprocessed gs://qtbtx-data/
```

### Step 2: Create VM and mount storage

```bash
# On VM, download data from GCS
gsutil -m cp -r gs://qtbtx-data/preprocessed ./data/
```

---

## Option 3: Cloud Run (Containerized)

### Step 1: Build and Push Docker Image

```bash
cd QTbtx

# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/qtbtx-optimizer -f cloud/Dockerfile .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/qtbtx-optimizer
```

### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy qtbtx-optimizer \
    --image gcr.io/YOUR_PROJECT_ID/qtbtx-optimizer \
    --platform managed \
    --region europe-west3 \
    --memory 4Gi \
    --timeout 3600 \
    --set-env-vars DATA_PATH=/data,OUTPUT_PATH=/output,SIGNALS=all
```

---

## Estimated Run Times

| Signal | Combinations | Est. Time (4 CPU) |
|--------|-------------|-------------------|
| MOUNT | 3 | ~10 sec |
| CLIMB | 3 | ~10 sec |
| ARROW | 31 | ~2 min |
| COLLECT | 31 | ~2 min |
| SOLID | 511 | ~30 min |
| RESIST | 511 | ~30 min |
| **ALL** | **1,090** | **~65 min** |

---

## Cost Estimate

- **e2-standard-4 VM**: ~$0.13/hour
- **1 hour run**: ~$0.15
- **Storage**: ~$0.02/GB/month

Total for full optimization: **< $0.50**

