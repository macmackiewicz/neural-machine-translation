import os

from boto3.session import Session
import sagemaker as sage


boto_sess = Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name='us-east-1'
)
sess = sage.Session(boto_session=boto_sess)
image = '081202375642.dkr.ecr.us-east-1.amazonaws.com/nmt:0.3.0-gpu'
hyperparameters = {
    'train_test_split': 0.001,
    'data_filename': 'deu-10k.txt'
}

clf = sage.estimator.Estimator(image_name=image,
                               role='sagemaker',
                               train_instance_count=1,
                               train_instance_type='ml.p2.xlarge',
                               hyperparameters=hyperparameters,
                               output_path="s3://neural-mt/output",
                               sagemaker_session=sess)

clf.fit({'training': 's3://neural-mt/data'})
