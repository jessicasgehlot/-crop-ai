#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# aws_deploy.sh — Deploy AI Crop Recommendation System on AWS EC2
#
# Usage:
#   1. Launch an EC2 instance (Amazon Linux 2 / Ubuntu 22.04, t2.micro or t3.small)
#   2. Open Security Group: port 22 (SSH), port 5000 (App), port 80 (optional)
#   3. SSH into instance and run:
#        chmod +x aws_deploy.sh && ./aws_deploy.sh
#
# OR paste this as EC2 User Data for automatic setup on launch.
# ─────────────────────────────────────────────────────────────────────────────

set -e

DOCKERHUB_IMAGE="jessicagehlot/crop-ai:latest"
CONTAINER_NAME="crop-ai"
APP_PORT="5000"

echo "=== [1/5] Updating system ==="
sudo yum update -y 2>/dev/null || sudo apt-get update -y

echo "=== [2/5] Installing Docker ==="
# Amazon Linux 2
if command -v yum &>/dev/null; then
    sudo yum install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ec2-user
# Ubuntu
else
    sudo apt-get install -y docker.io docker-compose
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ubuntu
fi

echo "=== [3/5] Pulling Docker image ==="
sudo docker pull ${DOCKERHUB_IMAGE}

echo "=== [4/5] Stopping old container (if any) ==="
sudo docker stop ${CONTAINER_NAME} 2>/dev/null || true
sudo docker rm   ${CONTAINER_NAME} 2>/dev/null || true

echo "=== [5/5] Starting application container ==="
sudo docker run -d \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    -p ${APP_PORT}:5000 \
    -e FLASK_DEBUG=false \
    -e PORT=5000 \
    --health-cmd="python -c \"import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')\"" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    --health-start-period=60s \
    ${DOCKERHUB_IMAGE}

echo ""
echo "=== Deployment complete ==="
echo "App running at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):${APP_PORT}"
echo "Health check:   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):${APP_PORT}/api/health"
echo ""

# Optional: start monitoring container
echo "=== Starting monitoring sidecar ==="
sudo docker run -d \
    --name crop-ai-monitor \
    --restart unless-stopped \
    --link ${CONTAINER_NAME}:crop-ai \
    -e APP_URL=http://crop-ai:5000 \
    -e CONTAINER_NAME=${CONTAINER_NAME} \
    ${DOCKERHUB_IMAGE} \
    sh -c "while true; do python monitoring/monitor.py --url http://crop-ai:5000; sleep 60; done"

echo "Monitoring container started."
echo ""
echo "=== Useful commands ==="
echo "  docker logs -f ${CONTAINER_NAME}          # view app logs"
echo "  docker logs -f crop-ai-monitor            # view monitor logs"
echo "  docker stats                              # live resource usage"
echo "  docker exec -it ${CONTAINER_NAME} bash    # shell into container"
