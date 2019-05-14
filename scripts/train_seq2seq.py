import os

from keras.callbacks import ModelCheckpoint, TensorBoard

from nmt.evaluators import Sequence2SequenceEvaluator
from nmt.readers import DelimitedTxtReader


def train(data_path: str, output_dir: str, train_test_split: float=0.2,
          verbose: int=2) -> Sequence2SequenceEvaluator:
    reader = DelimitedTxtReader(data_path)
    dataset = reader.get_dataset()

    evaluator = Sequence2SequenceEvaluator(dataset, train_test_split)

    checkpoint_dir = os.path.join(output_dir, evaluator.timestamp)

    checkpoint_path = '{}/model-{}.h5'.format(checkpoint_dir,
                                              evaluator.timestamp)
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                          verbose=verbose, save_best_only=True,
                                          mode='min')
    tensorboard_callback = TensorBoard(os.path.join(checkpoint_dir))

    evaluator.train(epochs=20, batch_size=64,
                    callbacks=[checkpoint_callback, tensorboard_callback],
                    verbose=verbose)

    return evaluator
