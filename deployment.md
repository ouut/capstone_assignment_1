# GCP High-Availability Deployment Guide

## FashionMNIST CNN Predictor — Production Deployment on Google Cloud

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Option A: Cloud Run (Recommended)](#option-a-cloud-run-recommended)
4. [Option B: Google Kubernetes Engine (GKE)](#option-b-google-kubernetes-engine-gke)
5. [API Key Authentication](#api-key-authentication)
6. [Monitoring & Observability](#monitoring--observability)
7. [Cost Estimation](#cost-estimation)
8. [Troubleshooting](#troubleshooting)
9. [Cleanup](#cleanup)

---

## Architecture Overview

```
                          ┌──────────────────────────────────┐
                          │        Google Cloud Platform      │
                          │                                   │
  Internet                │  ┌─────────────────────────────┐ │
  (HTTPS)                 │  │     Cloud Load Balancing     │ │
     │                    │  │   (Global, multi-region)     │ │
     ▼                    │  └──────────┬──────────────────┘ │
  ┌──────┐                │             │                    │
  │ API  │─── API Key ──▶ │  ┌──────────▼──────────────────┐ │
  │ Key  │   (Header)     │  │   Cloud Run / GKE Service  │ │
  └──────┘                │  │  ┌────────┐  ┌───────────┐  │ │
                          │  │  │ Pod 1  │  │  Pod 2..N │  │ │
                          │  │  │ (zone A│  │ (zone B)  │  │ │
                          │  │  └────────┘  └───────────┘  │ │
                          │  │   Auto-scaling (1–10 pods)  │ │
                          │  └─────────────────────────────┘ │
                          │                                   │
                          │  ┌─────────────────────────────┐ │
                          │  │  Cloud Monitoring / Logging  │ │
                          │  │  (metrics, alerts, traces)   │ │
                          │  └─────────────────────────────┘ │
                          └──────────────────────────────────┘
```

### Why This Architecture?

| Requirement | GCP Solution | HA Benefit |
|-------------|-------------|------------|
| Stateless API serving | Cloud Run (serverless) | Auto-scales per request; zero idle cost |
| Model baked in image | Artifact Registry | Immutable images; rollback in seconds |
| Multi-zone resilience | Cloud Run multi-region | Survives zonal outage automatically |
| Secure endpoint | API Gateway + API Key | Rate limiting, auth, TLS termination |
| Observability | Cloud Monitoring + Logging | Metrics, alerts, distributed tracing |
| CI/CD (optional) | Cloud Build | Auto-deploy on git push |

---

## Prerequisites

### 1. Google Cloud Account & Project

```bash
# Create a new project (or use existing)
gcloud projects create fashionmnist-deploy --name="FashionMNIST Deployment"

# Set as active project
gcloud config set project fashionmnist-deploy

# Link billing account (REQUIRED)
# Go to: https://console.cloud.google.com/billing
# Or via CLI:
gcloud billing accounts list
gcloud billing projects link fashionmnist-deploy --billing-account=BILLING_ACCOUNT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  container.googleapis.com \
  compute.googleapis.com
```

### 3. Install & Authenticate gcloud CLI

```bash
# Install (macOS)
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 4. Fix the Model Auto-Load Issue

Before deploying, the model must auto-load at server startup (assignment requirement). The current `server.py` has a `get_model()` function that raises an error if no model is loaded, but no `/upload-model` endpoint exists and the model isn't auto-loaded.

**Add this to `server.py`** after the global variable declarations (around line 49):

```python
# ---- Auto-load model at startup ----
import os

def _load_model_on_startup() -> None:
    """Load the trained model from MODEL_PATH at server startup."""
    global _model
    model_path = os.getenv("MODEL_PATH", "/app/models/best_model.pt")
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}. Set MODEL_PATH env var.")
        return
    try:
        _model = CNN()
        _model.load_state_dict(torch.load(model_path, map_location=_device, weights_only=True))
        _model.to(_device)
        _model.eval()
        print(f"Model loaded from {model_path} on {_device}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise

_load_model_on_startup()
```

---

## Option A: Cloud Run (Recommended)

Cloud Run is the **simplest path to high availability**. It's a fully-managed serverless platform that:

- Auto-scales each container instance (including to zero)
- Distributes across multiple zones automatically
- Provides HTTPS endpoint with managed TLS certificate
- Supports concurrency (multiple requests per container)
- Integrates with Cloud Monitoring out of the box

### Step 1: Create Artifact Registry Repository

```bash
# Create a Docker repository in us-central1
gcloud artifacts repositories create fashionmnist-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="FashionMNIST Docker images"
```

### Step 2: Build & Push Docker Image

```bash
# Set project ID variable
export PROJECT_ID=$(gcloud config get-value project)

# Build the image (using Cloud Build — no local Docker needed)
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/$PROJECT_ID/fashionmnist-repo/fashionmnist-server:v1 \
  .

# Or build locally then push (faster iterations):
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/fashionmnist-repo/fashionmnist-server:v1 .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/fashionmnist-repo/fashionmnist-server:v1
```

> **⚠️ Image Size Note:** The conda-based Docker image is ~3.7 GB. Cloud Run has a 4 GB image limit. If you hit the limit, see the **Image Optimization** section at the bottom of this guide.

### Step 3: Deploy to Cloud Run

```bash
gcloud run deploy fashionmnist-server \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/fashionmnist-repo/fashionmnist-server:v1 \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --concurrency 10 \
  --timeout 30 \
  --port 8000 \
  --set-env-vars "MODEL_PATH=/app/models/best_model.pt" \
  --allow-unauthenticated
```

| Flag | Value | Why |
|------|-------|-----|
| `--platform managed` | managed | Fully managed by Google (no cluster to maintain) |
| `--region us-central1` | us-central1 | Multi-zone region, good latency for US |
| `--memory 2Gi` | 2 GiB | PyTorch model needs ~1.5 GiB; 2 GiB gives headroom |
| `--cpu 2` | 2 vCPU | Helps with PyTorch inference speed |
| `--min-instances 1` | 1 | **Critical for HA**: keeps 1 instance always warm (no cold start) |
| `--max-instances 10` | 10 | Caps cost; auto-scales up to 10 under load |
| `--concurrency 10` | 10 | 10 concurrent requests per instance |
| `--timeout 30` | 30s | Inference takes <1s; 30s allows for cold start |
| `--port 8000` | 8000 | Matches uvicorn port in Dockerfile |
| `--allow-unauthenticated` | — | Public endpoint (use API key for auth — see below) |

### Step 4: Verify Deployment

```bash
# Get the service URL
gcloud run services describe fashionmnist-server \
  --region us-central1 \
  --format='value(status.url)'

# Test the health endpoint
curl -s https://YOUR-SERVICE-URL.a.run.app/health | jq

# Test prediction
curl -s -X POST https://YOUR-SERVICE-URL.a.run.app/predict-json \
  -H "Content-Type: application/json" \
  -d '{"image": [0]*784}' | jq

# Test class listing
curl -s https://YOUR-SERVICE-URL.a.run.app/classes | jq
```

### Step 5: Configure Multi-Region for Higher HA

For disaster recovery across entire regions, deploy to a **second region** and put a global load balancer in front:

```bash
# Deploy to second region
gcloud run deploy fashionmnist-server \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/fashionmnist-repo/fashionmnist-server:v1 \
  --platform managed \
  --region us-east1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated

# Create a serverless NEG (Network Endpoint Group) for each region
gcloud compute network-endpoint-groups create fashionmnist-neg-central \
  --region=us-central1 \
  --network-endpoint-type=serverless \
  --cloud-run-service=fashionmnist-server

gcloud compute network-endpoint-groups create fashionmnist-neg-east \
  --region=us-east1 \
  --network-endpoint-type=serverless \
  --cloud-run-service=fashionmnist-server

# Create backend service
gcloud compute backend-services create fashionmnist-backend \
  --global \
  --load-balancing-scheme=EXTERNAL_MANAGED

# Add NEGs to backend
gcloud compute backend-services add-backend fashionmnist-backend \
  --global \
  --network-endpoint-group=fashionmnist-neg-central \
  --network-endpoint-group-region=us-central1

gcloud compute backend-services add-backend fashionmnist-backend \
  --global \
  --network-endpoint-group=fashionmnist-neg-east \
  --network-endpoint-group-region=us-east1

# Create URL map and target proxy
gcloud compute url-maps create fashionmnist-url-map \
  --default-service=fashionmnist-backend

gcloud compute target-http-proxies create fashionmnist-http-proxy \
  --url-map=fashionmnist-url-map

# Create global forwarding rule (this gives you a single global IP)
gcloud compute forwarding-rules create fashionmnist-forwarding-rule \
  --global \
  --target-http-proxy=fashionmnist-http-proxy \
  --ports=80
```

> With this setup, if an entire region goes down, traffic automatically routes to the healthy region.

---

## Option B: Google Kubernetes Engine (GKE)

Use GKE when you need:
- More control over networking and pod scheduling
- GPU support (if you later want GPU inference)
- Custom horizontal pod autoscaling (HPA) metrics
- Sidecar containers (e.g., logging, proxy)

### Step 1: Create GKE Cluster

```bash
# Create a regional cluster (HA control plane, multi-zone nodes)
gcloud container clusters create fashionmnist-cluster \
  --region us-central1 \
  --num-nodes 1 \
  --min-nodes 1 \
  --max-nodes 5 \
  --machine-type e2-standard-2 \
  --enable-autoscaling \
  --enable-ip-alias \
  --workload-pool=$PROJECT_ID.svc.id.goog

# Get credentials
gcloud container clusters get-credentials fashionmnist-cluster --region us-central1
```

### Step 2: Create Kubernetes Manifests

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fashionmnist-server
  labels:
    app: fashionmnist
spec:
  replicas: 2                    # HA: always at least 2 replicas
  selector:
    matchLabels:
      app: fashionmnist
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0          # Zero-downtime deploys
      maxSurge: 1
  template:
    metadata:
      labels:
        app: fashionmnist
    spec:
      topologySpreadConstraints:  # Spread across zones
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: fashionmnist
      containers:
        - name: server
          image: us-central1-docker.pkg.dev/PROJECT_ID/fashionmnist-repo/fashionmnist-server:v1
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: "/app/models/best_model.pt"
          resources:
            requests:
              memory: "1.5Gi"
              cpu: "1"
            limits:
              memory: "2Gi"
              cpu: "2"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: fashionmnist-service
spec:
  type: LoadBalancer
  selector:
    app: fashionmnist
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fashionmnist-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fashionmnist-server
  minReplicas: 2               # HA minimum
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

> Replace `PROJECT_ID` with your actual GCP project ID.

### Step 3: Deploy to GKE

```bash
# Replace PROJECT_ID in the manifest
sed -i '' "s/PROJECT_ID/$PROJECT_ID/g" k8s/deployment.yaml

# Apply
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -w
kubectl get svc

# Get external IP
kubectl get svc fashionmnist-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Test
curl http://EXTERNAL_IP/health
```

### Step 4: Enable Managed TLS (Optional but Recommended)

```bash
# Using GKE Gateway + Managed Certificate
# Create k8s/managed-cert.yaml:
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: fashionmnist-cert
spec:
  domains:
    - api.yourdomain.com
EOF
```

---

## API Key Authentication

The assignment requires API key authentication. Implement this in `server.py`:

### 1. Add API Key Middleware

Add to `server.py` before the endpoint definitions:

```python
from fastapi import Security, status
from fastapi.security import APIKeyHeader
import os

# ---- API Key Authentication ----
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "change-me-in-production")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    """Validate API key from request header."""
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing. Include X-API-Key header.",
        )
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return key
```

### 2. Protect Endpoints

Change the endpoint decorators to require the dependency:

```python
@app.post("/predict-json")
async def predict_json(
    data: dict = Body(...),
    api_key: str = Security(verify_api_key),   # <-- add this
) -> dict[str, object]:
    ...
```

### 3. Set API Key as Secret

```bash
# Generate a strong key
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "Your API key: $API_KEY"
echo "Save this — it won't be shown again!"

# For Cloud Run — store as secret
echo -n "$API_KEY" | gcloud secrets create fashionmnist-api-key \
  --replication-policy="automatic" \
  --data-file=-

# Grant Cloud Run access to the secret
gcloud secrets add-iam-policy-binding fashionmnist-api-key \
  --member="serviceAccount:$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Redeploy Cloud Run with the secret
gcloud run deploy fashionmnist-server \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/fashionmnist-repo/fashionmnist-server:v1 \
  --region us-central1 \
  --set-secrets "API_KEY=fashionmnist-api-key:latest" \
  --memory 2Gi --cpu 2 --min-instances 1 --max-instances 10
```

### 4. Test Authenticated Requests

```bash
# Should FAIL (401)
curl -s -X POST https://YOUR-SERVICE-URL.a.run.app/predict-json \
  -H "Content-Type: application/json" \
  -d '{"image": [0]*784}'

# Should SUCCEED
curl -s -X POST https://YOUR-SERVICE-URL.a.run.app/predict-json \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"image": [0]*784}' | jq
```

---

## Monitoring & Observability

### Cloud Monitoring Dashboard

```bash
# Create a custom dashboard
gcloud monitoring dashboards create --config-from-file - <<EOF
displayName: FashionMNIST Dashboard
dashboardFilters:
  - filterType: RESOURCE_LABEL
    labelKey: service_name
    stringValue: fashionmnist-server
gridLayout:
  columns: "2"
  widgets:
    - title: "Request Latency (p50/p95/p99)"
      xyChart:
        dataSets:
          - timeSeriesQuery:
              timeSeriesFilter:
                filter: 'metric.type="run.googleapis.com/request_latencies" resource.type="cloud_run_revision"'
                aggregation:
                  perSeriesAligner: ALIGN_PERCENTILE_50
          - timeSeriesQuery:
              timeSeriesFilter:
                filter: 'metric.type="run.googleapis.com/request_latencies" resource.type="cloud_run_revision"'
                aggregation:
                  perSeriesAligner: ALIGN_PERCENTILE_95
          - timeSeriesQuery:
              timeSeriesFilter:
                filter: 'metric.type="run.googleapis.com/request_latencies" resource.type="cloud_run_revision"'
                aggregation:
                  perSeriesAligner: ALIGN_PERCENTILE_99
    - title: "Request Count"
      xyChart:
        dataSets:
          - timeSeriesQuery:
              timeSeriesFilter:
                filter: 'metric.type="run.googleapis.com/request_count" resource.type="cloud_run_revision"'
                aggregation:
                  perSeriesAligner: ALIGN_RATE
    - title: "Instance Count"
      xyChart:
        dataSets:
          - timeSeriesQuery:
              timeSeriesFilter:
                filter: 'metric.type="run.googleapis.com/container/instance_count" resource.type="cloud_run_revision"'
    - title: "CPU Utilization"
      xyChart:
        dataSets:
          - timeSeriesQuery:
              timeSeriesFilter:
                filter: 'metric.type="run.googleapis.com/container/cpu/utilizations" resource.type="cloud_run_revision"'
EOF
```

### Set Up Alerts

```bash
# Alert on high error rate
gcloud monitoring policies create \
  --display-name="FashionMNIST High Error Rate" \
  --condition-filter='metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class="5xx"' \
  --condition-threshold-value=5 \
  --condition-threshold-duration=300s \
  --notification-channels=YOUR_CHANNEL_ID

# Alert on high latency
gcloud monitoring policies create \
  --display-name="FashionMNIST High Latency" \
  --condition-filter='metric.type="run.googleapis.com/request_latencies"' \
  --condition-threshold-value=3000 \
  --condition-threshold-duration=300s \
  --notification-channels=YOUR_CHANNEL_ID
```

### Cloud Logging

All stdout/stderr from your FastAPI app flows automatically to Cloud Logging:

```bash
# View recent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=fashionmnist-server" \
  --limit=20 \
  --format="table(timestamp, textPayload)"

# Stream logs in real-time
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=fashionmnist-server"
```

---

## Cost Estimation

### Cloud Run (Recommended)

| Resource | Spec | Monthly Cost (approx) |
|----------|------|----------------------|
| 1 always-warm instance | 2 vCPU, 2 GiB | ~$68/mo |
| Additional request-driven instances | 2 vCPU, 2 GiB | ~$0.000054/sec ($0.19/hr) |
| Artifact Registry storage | ~4 GB | ~$0.40/mo |
| Cloud Monitoring | Basic metrics | Free tier |
| **Total (low traffic)** | — | **~$70/mo** |
| **Total (moderate, ~10 req/s)** | 2–3 instances avg | **~$90–120/mo** |

### GKE (Alternative)

| Resource | Spec | Monthly Cost (approx) |
|----------|------|----------------------|
| GKE cluster (regional) | Control plane fee | $73/mo |
| 2 nodes (HA minimum) | e2-standard-2 | ~$97/mo |
| Load Balancer | Global | ~$18/mo |
| Artifact Registry | ~4 GB | ~$0.40/mo |
| **Total (GKE)** | — | **~$188/mo** |

> 💡 **Recommendation**: Cloud Run is significantly cheaper and simpler for this use case. GKE only makes sense if you need GPU inference or have complex networking requirements.

### Free Tier Credits

New GCP accounts get **$300 free credits** for 90 days, which covers this entire deployment and then some.

---

## Image Optimization (If Needed)

The conda-based image is ~3.7 GB. Cloud Run caps at 4 GB. If you're close to the limit, switch to a pip-based multi-stage build:

```dockerfile
# ---- Optimized Dockerfile (pip-based, ~800 MB) ----
FROM python:3.12-slim AS builder

WORKDIR /app
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    torchvision --index-url https://download.pytorch.org/whl/cpu \
    fastapi uvicorn pillow numpy

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libjpeg62-turbo-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

WORKDIR /app
COPY server.py cnn.py dnn.py ./
COPY models/best_model.pt /app/models/best_model.pt

ENV MODEL_PATH=/app/models/best_model.pt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

> This optimized image is ~800 MB vs 3.7 GB — faster deploy, lower storage costs, well within Cloud Run limits.

---

## Troubleshooting

### Container fails to start (CrashLoopBackOff / Cloud Run error)

```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=20

# Common causes:
# 1. Model file not found at MODEL_PATH
# 2. PyTorch version mismatch
# 3. OOM (out of memory) — increase --memory flag
```

### Cold start latency too high (>5 seconds)

```bash
# Set min-instances=1 to eliminate cold starts
gcloud run services update fashionmnist-server --min-instances 1

# Or reduce image size (see optimization section above)
```

### 503 errors under load

```bash
# Increase max instances
gcloud run services update fashionmnist-server --max-instances 20

# Increase concurrency
gcloud run services update fashionmnist-server --concurrency 20

# Check if hitting memory limits → increase --memory
```

### API key rejected

```bash
# Verify secret is accessible
gcloud secrets versions access latest --secret=fashionmnist-api-key

# Check the key was set in the environment
gcloud run services describe fashionmnist-server --format='yaml(spec.template.spec.containers[0].env)'
```

---

## Deployment Checklist

Before submitting the assignment, verify:

- [ ] `server.py` auto-loads model from `MODEL_PATH` at startup
- [ ] Docker image builds and runs locally (`./docker-build.sh`)
- [ ] Image pushed to Artifact Registry
- [ ] Cloud Run (or GKE) deployed and health check passes
- [ ] `/predict-json` returns valid predictions
- [ ] API key authentication working (401 without key, 200 with key)
- [ ] Minimum 2 instances (or `min-instances=1`) for HA
- [ ] Cloud Monitoring dashboard created
- [ ] Alert policies configured
- [ ] Screenshots taken:
  - Cloud Run service details page
  - Successful curl request with prediction
  - Monitoring dashboard showing metrics
  - Cloud Logging showing request logs
- [ ] Service URL documented

---

## References

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Artifact Registry](https://cloud.google.com/artifact-registry/docs)
- [Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [Secret Manager](https://cloud.google.com/secret-manager/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
