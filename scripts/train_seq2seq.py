from datetime import datetime

from keras.callbacks import ModelCheckpoint

from nmt.evaluators import Sequence2SequenceEvaluator
from nmt.readers import DelimitedTxtReader


def train(data_path: str, checkpoint_dir: str, train_test_split: float=0.2,
          verbose: int=2) -> Sequence2SequenceEvaluator:
    reader = DelimitedTxtReader(data_path)
    dataset = reader.get_dataset()

    evaluator = Sequence2SequenceEvaluator(dataset, train_test_split)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    checkpoint_path = '{}/model-{}.h5'.format(checkpoint_dir, timestamp)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                 verbose=verbose, save_best_only=True,
                                 mode='min')

    evaluator.train(epochs=20, batch_size=64, callbacks=[checkpoint],
                    verbose=verbose)

    return evaluator
