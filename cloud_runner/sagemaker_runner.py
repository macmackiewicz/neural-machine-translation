import os

import sagemaker as sage
from dotenv import load_dotenv
from boto3.session import Session

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

region = os.environ.get('AWS_REGION', 'us-east-1')
role = os.environ.get('SAGEMAKER_ROLE', 'sagemaker')
instance_type = os.environ.get('INSTANCE_TYPE', 'ml.p2.xlarge')
docker_registry = os.environ.get('DOCKER_REGISTRY')
s3_bucket = os.environ.get('S3_BUCKET')

boto_sess = Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name=region
)
sess = sage.Session(boto_session=boto_sess)
image = '{}/nmt:latest-gpu'.format(docker_registry)


def sagemaker_train(config, wait=False):
    clf = sage.estimator.Estimator(image_name=image,
                                   role=role,
                                   train_instance_count=1,
                                   train_instance_type=instance_type,
                                   hyperparameters=config,
                                   output_path='s3://{}/output'.format(s3_bucket),
                                   sagemaker_session=sess)

    clf.fit({'training': 's3://{}/data'.format(s3_bucket)}, wait=wait)
