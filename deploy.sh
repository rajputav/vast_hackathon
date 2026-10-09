#!/usr/bin/env bash
# Deploy app/ to the team's Kubernetes namespace at https://<team host>/app
# (no Docker build: public python image + code from a ConfigMap + creds from a Secret).
# Re-run after any code change; it updates the ConfigMap and restarts the pod.
set -euo pipefail

cd "$(dirname "$0")"
APP_DIR=app
APP_NAME=${APP_NAME:-vss-app}
APP_PORT=8080

mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 -type f -name '*.config' | sort)
(( ${#TEAM_CONFIGS[@]} == 1 )) || { echo "expected exactly one /config/*.config" >&2; exit 1; }
TEAM_CONFIG="${TEAM_CONFIGS[0]}"
cfg() { grep "^$1=" "$TEAM_CONFIG" | cut -d= -f2-; }

NS=$(cfg USERNAME)
export KUBECONFIG=/config/${NS}-k8s.yaml
APP_HOST="video-lab-team-${NS#team-}.cosmos.vastdata.com"

# The team INGRESS_URL hostname doesn't resolve inside the cluster, so the pod talks to
# the VSS backend through its in-namespace Service instead (same API, same login).

# W&B keys live in the VM environment (/etc/environment), not the team config.
if [[ -z "${WANDB_API_KEY:-}" && -f /etc/environment ]]; then
  set -a; source /etc/environment; set +a
fi
[[ -n "${WANDB_API_KEY:-}" ]] || echo "warning: WANDB_API_KEY not set; the W&B check will fail" >&2

echo "==> namespace $NS, host $APP_HOST, app $APP_NAME"

kubectl -n "$NS" create configmap "${APP_NAME}-code" --from-file="$APP_DIR" \
  --dry-run=client -o yaml | kubectl apply -f -

# court/ is a package; --from-file on a directory is flat and would also sweep up the
# ledger/report outputs, so mount only its .py files as a second ConfigMap at /pkg/court.
COURT_FILES=()
for f in court/*.py; do COURT_FILES+=(--from-file="$f"); done
kubectl -n "$NS" create configmap "${APP_NAME}-court" "${COURT_FILES[@]}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" create secret generic "${APP_NAME}-creds" \
  --from-literal=VSS_URL="http://video-backend-service:8000" \
  --from-literal=VSS_USERNAME="$NS" \
  --from-literal=VSS_PASSWORD="$(cfg PASSWORD)" \
  --from-literal=COSMOS_URL="http://166.19.38.112:8001" \
  --from-literal=COSMOS_TOKEN="$(cfg GPU_BEARER_TOKEN)" \
  --from-literal=WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --from-literal=WANDB_PROJECT="${WANDB_TEAM:-}/${WANDB_PROJECT:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  replicas: 1
  selector:
    matchLabels: {app: ${APP_NAME}}
  template:
    metadata:
      labels: {app: ${APP_NAME}}
    spec:
      containers:
      - name: app
        image: python:3.12-slim
        imagePullPolicy: IfNotPresent
        ports: [{containerPort: ${APP_PORT}}]
        env:
        - {name: PORT, value: "${APP_PORT}"}
        - {name: COURT_OUT_DIR, value: /tmp/court}   # ConfigMap mounts are read-only
        - {name: PYTHONPATH, value: /pkg}            # /pkg/court is the court package
        envFrom: [{secretRef: {name: ${APP_NAME}-creds}}]
        volumeMounts:
        - {name: code, mountPath: /code}
        # Not nested under /code: the ConfigMap volume writer owns that directory.
        - {name: court, mountPath: /pkg/court}
        workingDir: /code
        command: ["bash", "-c"]
        args:
        - |
          set -euo pipefail
          pip install --no-cache-dir -q --disable-pip-version-check --root-user-action=ignore -r requirements.txt
          exec python main.py
        readinessProbe:
          httpGet: {path: /health, port: ${APP_PORT}}
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests: {cpu: 250m, memory: 512Mi}
          limits: {memory: 2Gi}
      volumes:
      - name: code
        configMap: {name: ${APP_NAME}-code}
      - name: court
        configMap: {name: ${APP_NAME}-court}
---
apiVersion: v1
kind: Service
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  selector: {app: ${APP_NAME}}
  ports: [{name: http, port: 80, targetPort: ${APP_PORT}}]
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  ingressClassName: nginx
  rules:
  - host: ${APP_HOST}
    http:
      paths:
      - path: /app(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: ${APP_NAME}
            port: {number: 80}
EOF

# ConfigMap changes aren't picked up by a running pod.
kubectl -n "$NS" rollout restart deploy/"$APP_NAME"
kubectl -n "$NS" rollout status deploy/"$APP_NAME" --timeout=240s

echo "==> health: $(curl -sS -m 10 "http://${APP_HOST}/app/health" || echo unreachable)"
echo "Open https://workshop.thecosmoslabs.com and click the App button."
