#!/usr/bin/env bash
#
# Deploy vectorizer API to AWS Lambda via ECR.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker running
#   - GEMINI_API_KEY set in environment (or passed as argument)
#
# Usage:
#   bash deploy.sh                          # uses GEMINI_API_KEY from env
#   GEMINI_API_KEY=abc123 bash deploy.sh    # explicit key
#
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
FUNCTION_NAME="vectorizer-api"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${FUNCTION_NAME}"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest"
MEMORY=1024        # MB
TIMEOUT=300        # seconds (5 min)
EPHEMERAL_STORAGE=1024  # MB for /tmp

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set"
    echo "  export GEMINI_API_KEY=your-key-here"
    exit 1
fi

echo "=== Vectorizer API Deploy ==="
echo "  Region:   ${REGION}"
echo "  Account:  ${ACCOUNT_ID}"
echo "  Function: ${FUNCTION_NAME}"
echo ""

# ── 1. Create ECR repository (if needed) ─────────────────────
echo "[1/5] Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" --image-scanning-configuration scanOnPush=true

# ── 2. Build Docker image ────────────────────────────────────
echo "[2/5] Building Docker image..."
cd "$(dirname "$0")"
docker build --platform linux/amd64 -t "${ECR_REPO}:latest" .

# ── 3. Push to ECR ───────────────────────────────────────────
echo "[3/5] Pushing to ECR..."
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker tag "${ECR_REPO}:latest" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

# ── 4. Create or update Lambda function ──────────────────────
echo "[4/5] Creating/updating Lambda function..."

# Check if function exists
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" 2>/dev/null; then
    # Update existing function
    aws lambda update-function-code \
        --function-name "${FUNCTION_NAME}" \
        --image-uri "${IMAGE_URI}" \
        --region "${REGION}"

    # Wait for update to complete before changing config
    aws lambda wait function-updated --function-name "${FUNCTION_NAME}" --region "${REGION}"

    aws lambda update-function-configuration \
        --function-name "${FUNCTION_NAME}" \
        --memory-size "${MEMORY}" \
        --timeout "${TIMEOUT}" \
        --ephemeral-storage "Size=${EPHEMERAL_STORAGE}" \
        --environment "Variables={GEMINI_API_KEY=${GEMINI_API_KEY}}" \
        --region "${REGION}"
else
    # Create execution role if needed
    ROLE_NAME="${FUNCTION_NAME}-role"
    ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

    if ! aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
        echo "  Creating execution role: ${ROLE_NAME}"
        aws iam create-role \
            --role-name "${ROLE_NAME}" \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }'
        aws iam attach-role-policy \
            --role-name "${ROLE_NAME}" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        echo "  Waiting for role propagation..."
        sleep 10
    fi
    ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)

    aws lambda create-function \
        --function-name "${FUNCTION_NAME}" \
        --package-type Image \
        --code "ImageUri=${IMAGE_URI}" \
        --role "${ROLE_ARN}" \
        --memory-size "${MEMORY}" \
        --timeout "${TIMEOUT}" \
        --ephemeral-storage "Size=${EPHEMERAL_STORAGE}" \
        --environment "Variables={GEMINI_API_KEY=${GEMINI_API_KEY}}" \
        --region "${REGION}"

    aws lambda wait function-active --function-name "${FUNCTION_NAME}" --region "${REGION}"
fi

# ── 5. Enable Function URL ───────────────────────────────────
echo "[5/5] Enabling Function URL..."
FUNCTION_URL=$(aws lambda get-function-url-config \
    --function-name "${FUNCTION_NAME}" \
    --region "${REGION}" \
    --query 'FunctionUrl' --output text 2>/dev/null || true)

if [ -z "${FUNCTION_URL}" ] || [ "${FUNCTION_URL}" = "None" ]; then
    FUNCTION_URL=$(aws lambda create-function-url-config \
        --function-name "${FUNCTION_NAME}" \
        --auth-type NONE \
        --region "${REGION}" \
        --query 'FunctionUrl' --output text)

    # Allow public access
    aws lambda add-permission \
        --function-name "${FUNCTION_NAME}" \
        --statement-id "FunctionURLAllowPublicAccess" \
        --action "lambda:InvokeFunctionUrl" \
        --principal "*" \
        --function-url-auth-type NONE \
        --region "${REGION}" 2>/dev/null || true
fi

echo ""
echo "=== Deploy Complete ==="
echo "  Function URL: ${FUNCTION_URL}"
echo ""
echo "  Test:"
echo "    curl -F 'file=@test.jpeg' ${FUNCTION_URL}vectorize -o result.zip"
echo "    unzip -l result.zip"
