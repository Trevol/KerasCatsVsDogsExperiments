from model import makeModel
from dataset import makeDataset


def train():
    inputSize = (256, 256)
    model = makeModel(inputSize, compileForTraining=True, weights=None)

    epochs = 50
    batch_size = 2

    train_generator, validation_generator = makeDataset('dataset', inputSize, batch_size)

    nb_train_samples = train_generator.samples
    nb_validation_samples = validation_generator.samples

    model.fit_generator(
        train_generator,
        steps_per_epoch=1000,  # nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('train_session_1.h5')


if __name__ == '__main__':
    train()
