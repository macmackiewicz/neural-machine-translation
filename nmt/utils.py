import os

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from nmt.evaluators import Sequence2SequenceEvaluator
from nmt.readers import DelimitedTxtReader


def train_seq2seq(data_path: str, report_dir: str,
                  train_validation_split: float=0.2,
                  train_test_split: float=0.0,
                  epochs: int=15, batch_size: int=64, **hyperparameters) \
        -> Sequence2SequenceEvaluator:
    reader = DelimitedTxtReader(data_path)
    dataset = reader.get_dataset()

    evaluator = Sequence2SequenceEvaluator(dataset, train_validation_split,
                                           train_test_split,
                                           **hyperparameters)

    output_dir = os.path.join(report_dir, evaluator.timestamp)

    checkpoint_path = '{}/checkpoint.h5'.format(output_dir,
                                                evaluator.timestamp)
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                          verbose=2, save_best_only=True,
                                          mode='min')

    stopping_callback = EarlyStopping(monitor='val_loss', patience=2,
                                      mode='min', verbose=1)

    tensorboard_callback = TensorBoard(os.path.join(output_dir))

    evaluator.train(batch_size, epochs=epochs, verbose=2, callbacks=[
        checkpoint_callback, tensorboard_callback, stopping_callback
    ])

    return evaluator
