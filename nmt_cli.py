import json

import click

from nmt.readers import DelimitedTxtReader
from scripts.train_seq2seq import train as train_s2s
from nmt.evaluators import Sequence2SequenceEvaluator
from cloud_runner.sagemaker_runner import sagemaker_train


def get_evaluator(data_path: str, model_weights_path: str,
                  train_test_split: float) -> Sequence2SequenceEvaluator:
    reader = DelimitedTxtReader(data_path)
    dataset = reader.get_dataset()

    evaluator = Sequence2SequenceEvaluator(dataset, train_test_split)

    evaluator.model.load_weights(model_weights_path)

    return evaluator


@click.group()
def main():
    pass


@main.command()
@click.option('--data-path', '-d')
@click.option('--checkpoint-dir', '-c', default='./checkpoints')
@click.option('--train-test-split', default=0.2)
@click.option('--verbose', '-V', default=2)
def train(data_path, checkpoint_dir, train_test_split, verbose):
    click.echo('Let\'s 🚆')
    train_s2s(data_path, checkpoint_dir, train_test_split, verbose)


@main.command()
@click.option('--config-path', '-c', default='./config/default.json')
@click.option('--wait/--no-wait', default=False)
def sage_train(config_path, wait):
    with open(config_path, 'r') as f:
        config = json.load(f)

    click.echo('Training in the ☁️')
    sagemaker_train(config, wait)


@main.command()
@click.option('--data-path', '-d')
@click.option('--model-weights-path', '-m')
@click.option('--train-test-split', default=0.2)
def evaluate(data_path, model_weights_path, train_test_split):
    evaluator = get_evaluator(data_path, model_weights_path, train_test_split)

    click.echo('Evaluating...')
    click.echo(evaluator.get_bleu_score())


@main.command()
@click.option('--data-path', '-d')
@click.option('--model-weights-path', '-m')
@click.option('--train-test-split', default=0.2)
def translate(data_path, model_weights_path, train_test_split):
    evaluator = get_evaluator(data_path, model_weights_path, train_test_split)

    while True:
        sentence = input('provide a sentence in German:\n')
        click.echo('Translation:', evaluator.predict_sentence(sentence))


def cli():
    return main()
