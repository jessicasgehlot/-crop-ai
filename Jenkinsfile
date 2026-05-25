// ─────────────────────────────────────────────────────────────────────────────
// Jenkinsfile — AI Crop Recommendation System CI/CD Pipeline
//
// Stages:
//   1. Checkout        — pull source from GitHub
//   2. Install         — pip install dependencies
//   3. Test            — pytest with coverage
//   4. Build Image     — docker build
//   5. Push to Hub     — docker push to Docker Hub
//   6. Deploy          — docker-compose up on target server
//   7. Health Check    — run monitoring script
//
// Required Jenkins Credentials:
//   dockerhub-credentials  → Docker Hub username/password
//   github-token           → GitHub personal access token (for webhooks)
//
// Required Jenkins Plugins:
//   Pipeline, Git, Docker Pipeline, Credentials Binding, JUnit
// ─────────────────────────────────────────────────────────────────────────────

pipeline {

    agent any

    environment {
        DOCKER_IMAGE   = "jessicagehlot/crop-ai"
        DOCKER_TAG     = "${env.BUILD_NUMBER}"
        DOCKER_LATEST  = "latest"
        CONTAINER_NAME = "crop-ai"
        APP_PORT       = "5000"
        PYTHON         = "python"
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: "10"))
        timeout(time: 30, unit: "MINUTES")
        timestamps()
    }

    triggers {
        // Auto-trigger on GitHub push (requires GitHub webhook → Jenkins)
        githubPush()
    }

    stages {

        // ── Stage 1: Checkout ─────────────────────────────────────────────
        stage("Checkout") {
            steps {
                echo "Checking out source code..."
                checkout scm
                // Windows-safe
                bat "git log --oneline -5"
            }
        }

        // ── Stage 2: Install Dependencies ────────────────────────────────
        stage("Install Dependencies") {
            steps {
                echo "Installing Python dependencies..."
                bat "${PYTHON} -m pip install --upgrade pip"
                bat "${PYTHON} -m pip install -r requirements.txt"
            }
        }

        // ── Stage 3: Run Tests ────────────────────────────────────────────
        stage("Run Tests") {
            steps {
                echo "Running pytest test suite..."
                bat "${PYTHON} -m pytest tests/ -v --cov=. --cov-report=xml:coverage.xml --cov-report=term-missing --junitxml=test-results.xml --tb=short"
            }
            post {
                always {
                    junit "test-results.xml"
                    archiveArtifacts artifacts: "coverage.xml", allowEmptyArchive: true
                }
                failure {
                    echo "Tests FAILED — aborting pipeline"
                }
            }
        }

        // ── Stage 4: Build Docker Image ───────────────────────────────────
        stage("Build Docker Image") {
            steps {
                echo "Building Docker image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
                bat "docker build --tag ${DOCKER_IMAGE}:${DOCKER_TAG} --tag ${DOCKER_IMAGE}:${DOCKER_LATEST} --label \"build=${env.BUILD_NUMBER}\" --label \"commit=${env.GIT_COMMIT}\" ."
                bat "docker images ${DOCKER_IMAGE}"
            }
        }

        // ── Stage 5: Push to Docker Hub ────────────────────────────────────
        stage("Push to Docker Hub") {
            steps {
                echo "Pushing image to Docker Hub..."
                withCredentials([usernamePassword(
                    credentialsId: "dockerhub-credentials",
                    usernameVariable: "DOCKER_USER",
                    passwordVariable: "DOCKER_PASS"
                )]) {
                    bat "echo ${DOCKER_PASS} | docker login -u ${DOCKER_USER} --password-stdin"
                    bat "docker push ${DOCKER_IMAGE}:${DOCKER_TAG}"
                    bat "docker push ${DOCKER_IMAGE}:${DOCKER_LATEST}"
                    bat "docker logout"
                }
            }
        }

        // ── Stage 6: Deploy ───────────────────────────────────────────────
        stage("Deploy") {
            steps {
                echo "Deploying with docker-compose..."
                bat "docker-compose down --remove-orphans"
                bat "docker-compose pull"
                bat "docker-compose up -d --build"
                bat "docker-compose ps"
            }
        }

        // ── Stage 7: Health Check ─────────────────────────────────────────
        stage("Health Check") {
            steps {
                echo "Waiting for app to start..."
                bat "timeout /t 15 /nobreak"
                echo "Running monitoring health check..."
                bat "${PYTHON} monitoring/monitor.py --url http://localhost:${APP_PORT}"
            }
        }

    }

    // ── Post Actions ──────────────────────────────────────────────────────────
    post {
        success {
            echo """
            ╔══════════════════════════════════════╗
            ║  Pipeline SUCCESS                    ║
            ║  Image: ${DOCKER_IMAGE}:${DOCKER_TAG} 
            ║  Build: ${env.BUILD_NUMBER}          ║
            ╚══════════════════════════════════════╝
            """
        }
        failure {
            echo "Pipeline FAILED — check logs above"
        }
        always {
            bat "docker image prune -f"
            cleanWs()
        }
    }
}

