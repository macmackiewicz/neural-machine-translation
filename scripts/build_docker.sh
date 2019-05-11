set -e

export IMAGE_TYPE=$1

# export variables from .env file
export $(cat .env | xargs)
BASE_IMAGE_URI=${DOCKER_REGISTRY}/nmt

echo "Logging in to AWS ECR..."
aws ecr get-login --no-include-email --region ${AWS_REGION} | sh

NMT_VERSION=$(python setup.py --version)
echo "Current version: ${NMT_VERSION}"

if [ "${IMAGE_TYPE}" = "cpu" ]; then
    echo "Using cpu version of docker image"
    DOCKERFILE=Dockerfile
    IMAGE_URI=${BASE_IMAGE_URI}:${NMT_VERSION}
    IMAGE_URI_LATEST=${BASE_IMAGE_URI}:latest
else
    echo "Using gpu version of docker image"
    DOCKERFILE=Dockerfile.gpu
    IMAGE_URI=${BASE_IMAGE_URI}:${NMT_VERSION}-gpu
    IMAGE_URI_LATEST=${BASE_IMAGE_URI}:latest-gpu
fi

echo "Building docker image..."
docker build -t ${IMAGE_URI} -f ${DOCKERFILE} .
docker tag ${IMAGE_URI} ${IMAGE_URI_LATEST}

echo "Pushing docker image..."
docker push ${IMAGE_URI}
docker push ${IMAGE_URI_LATEST}

echo "Done!"
