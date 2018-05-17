import datetime
import keras
import tensorflow as tf


def model(sentiment_data):
    model = keras.models.Sequential([
        keras.layers.Embedding(sentiment_data.vocab_size, 100),
        keras.layers.Bidirectional(keras.layers.GRU(500, dropout=0.2, recurrent_dropout=0.2)),
        keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def main(config, training_set, testing_set, sentiment_data):
    sentiment_model = model(sentiment_data)

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    saver = keras.callbacks.ModelCheckpoint('./builds/' + timestamp + '/sentiment_checkpoint_epoch-{epoch:02d}.hdf5',
                                            monitor='val_loss', verbose=verbose, save_best_only=True)

    sentiment_model.fit_generator(sentiment_data.get_batch(), steps_per_epoch=len(sentiment_data) / config.batch_size,
                                  epochs=config.n_epochs,
                                  verbose=verbose,
                                  validation_data=sentiment_data.get_batch(test=True),
                                  validation_steps=sentiment_data.test_length() / config.batch_size,
                                  callbacks=[tensorboard, saver])
