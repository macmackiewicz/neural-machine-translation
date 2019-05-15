import json

import click

from nmt.utils import train_seq2seq
from nmt.readers import DelimitedTxtReader
from nmt.evaluators import Sequence2SequenceEvaluator
from cloud_runner.sagemaker_runner import sagemaker_train


def get_evaluator(data_path: str, model_weights_path: str,
                  train_test_split: float) -> Sequence2SequenceEvaluator:
    reader = DelimitedTxtReader(data_path)
    dataset = reader.get_dataset()

    return Sequence2SequenceEvaluator\
        .reconstruct_from_weights(dataset, model_weights_path, train_test_split)


@click.group()
def main():
    pass


@main.command()
@click.option('--data-path', '-d', type=click.STRING)
@click.option('--report-dir', '-r', default='./reports', type=click.STRING)
@click.option('--train-test-split', default=0.2, type=click.FLOAT)
@click.option('--verbose', '-V', default=2, type=click.INT)
def train(data_path: str, report_dir: str, train_test_split: float,
          verbose: int):
    click.echo('Let\'s 🚆')
    train_seq2seq(data_path, report_dir, train_test_split, verbose)


@main.command()
@click.option('--config-path', '-c', default='./config/default.json',
              type=click.STRING)
@click.option('--wait/--no-wait', default=False)
def sage_train(config_path: str, wait: bool):
    with open(config_path, 'r') as f:
        config = json.load(f)

    click.echo('Training in the ☁️')
    sagemaker_train(config, wait)


@main.command()
@click.option('--data-path', '-d', type=click.STRING)
@click.option('--model-weights-path', '-m', type=click.STRING)
@click.option('--train-test-split', default=0.2, type=click.FLOAT)
def evaluate(data_path: str, model_weights_path: str, train_test_split: float):
    evaluator = get_evaluator(data_path, model_weights_path, train_test_split)

    click.echo('Evaluating...')
    click.echo(evaluator.get_bleu_score())


@main.command()
@click.option('--data-path', '-d', type=click.STRING)
@click.option('--model-weights-path', '-m', type=click.STRING)
@click.option('--train-test-split', default=0.2, type=click.FLOAT)
def translate(data_path: str, model_weights_path: str, train_test_split: float):
    evaluator = get_evaluator(data_path, model_weights_path, train_test_split)

    while True:
        sentence = input('Provide a sentence in source language:\n')
        predicted_sentence = evaluator.predict_sentence(sentence)
        click.secho('Translation: {}'.format(predicted_sentence, fg='green'))


def cli():
    return main()
