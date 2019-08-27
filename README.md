# Neural Machine Translation

This project has been developed as a part of my master's thesis about 
Neural Machine Translations. 

## Dataset
Bilingual datasets can be found on [ManyThings.org](http://www.manythings.org/anki).  
Language pair used in this project is [deu-eng](http://www.manythings.org/anki/deu-eng.zip). 

## Running locally
Required python version: 3.5.2.

Install dependencies:
```commandline
pip install -r requirements-dev.txt
python setup.py install
```

Available CLI commands can be listed through `nmt --help`.

For example: `nmt train -d ./data/deu.txt` will train Sequence to Sequence model
using text data provided under given path.

## Running with docker
Build the image: `docker build -t nmt -f ./docker/Dockerfile .`.

Docker image exposes commands from the CLI, so they can be invoked 
the same way as locally, as long as volumes for data and output 
are properly mounted, e.g.:
```
docker run --rm -it -v ${PWD}/data:/opt/ml/input/data/training \
  -v ${PWD}/reports:/opt/ml/model \ 
  nmt train -d /opt/ml/input/data/training/deu.txt -r /opt/ml/model
```

## Training on AWS Sagemaker
Training the full dataset locally requires significant number of hours,
therefore it's advised to offload this process to the cloud.
Cloud runner of choice for this project is AWS Sagemaker.  
  
After setting up Sagemaker IAM role, S3 bucket and ECR registry, 
all related configuration should be placed in `.env` file 
structured in a similar way as `.env-template`, i.e.:
```
SAGEMAKER_ROLE=sagemaker
DOCKER_REGISTRY=
S3_BUCKET=
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
INSTANCE_TYPE=ml.p2.xlarge
```

Training command can be invoked with `nmt sage-train`, 
which accepts optional parameter (`-c`) specifying location of configuration file.
This configuration will be passed as hyperparameters to the training image.

## Using final model for translation
Trained model can be prompted to translate sentences through `nmt translate` command,
which requires two arguments to properly reconstruct the model:
- path to trained weights (`--model-weights`)
- path to data file used for training (`--data-path`)  
Data file is required to reconstruct tokenizer with vocabulary that allows
transformations of input sentences into vectors and vectors produced by the model
into output sentences.  

Example translations produced by the model:
![](https://s3.eu-central-1.amazonaws.com/nmt-public/static/deu-translation-example.png)

During the course of my work on the master's thesis I will be uploading
best performing models to public S3 bucket, so they can be used without the need
of training them from scratch.
- [model link](https://s3.eu-central-1.amazonaws.com/nmt-public/model/model.h5)
- [dataset link](https://s3.eu-central-1.amazonaws.com/nmt-public/model/deu.txt)

## Tensorboard
Model training metrics can be monitored with tensorboard: 
`tensorboard --logdir ./reports`
