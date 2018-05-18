import datetime
import keras
import os


def model(sentiment_data):
    model = keras.models.Sequential([
        keras.layers.Embedding(sentiment_data.vocab_size, 5),
        keras.layers.GRU(5, dropout=0.5, recurrent_dropout=0.5),
        keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def main(config, sentiment_data):
    sentiment_model = model(sentiment_data)

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/sentiment-' + timestamp + '/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    model_path = os.path.abspath(
        os.path.join(os.curdir, './builds/' + timestamp + '-'))

    model_path += 'sentiment_checkpoint_epoch-{epoch:02d}.hdf5'
    saver = keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_acc', verbose=verbose, save_best_only=True)

    sentiment_model.fit_generator(sentiment_data.get_batch(), steps_per_epoch=len(sentiment_data) / config.batch_size,
                                  epochs=config.n_epochs,
                                  verbose=verbose,
                                  validation_data=sentiment_data.get_batch(test=True),
                                  validation_steps=sentiment_data.test_length() / config.batch_size,
                                  callbacks=[tensorboard, saver])
