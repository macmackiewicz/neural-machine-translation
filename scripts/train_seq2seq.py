import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from sagemaker.utils import sagemaker_timestamp

from nmt.evaluators import Sequence2SequenceEvaluator
from nmt.readers import DelimitedTxtReader


def train(data_path: str, checkpoint_dir: str, train_test_split: float=0.2,
          verbose: int=2) -> Sequence2SequenceEvaluator:
    reader = DelimitedTxtReader(data_path)
    dataset = reader.get_dataset()

    evaluator = Sequence2SequenceEvaluator(dataset, train_test_split)

    timestamp = sagemaker_timestamp()

    checkpoint_path = '{}/model-{}.h5'.format(checkpoint_dir, timestamp)
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                          verbose=verbose, save_best_only=True,
                                          mode='min')
    tensorboard_callback = TensorBoard(os.path.join(checkpoint_dir,
                                                    'nmt-{}'.format(timestamp)))

    evaluator.train(epochs=20, batch_size=64,
                    callbacks=[checkpoint_callback, tensorboard_callback],
                    verbose=verbose)

    return evaluator
