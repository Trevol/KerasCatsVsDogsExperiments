from model import makeModel
from dataset import makeDataset
import keras
import os

def train():
    inputSize = (256, 256)
    model = makeModel(inputSize, compileForTraining=True, weights=None)

    epochs = 50
    batch_size = 20

    train_augmentations = dict(rescale=1. / 255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               rotation_range=5,
                               width_shift_range=7,
                               height_shift_range=7
                               )
    train_generator, validation_generator = makeDataset('dataset', inputSize, batch_size)

    nb_train_samples = train_generator.samples
    nb_validation_samples = validation_generator.samples

    os.makedirs('/mnt/HDD/training_checkpoints/KerasCatsVsDogsExperiments/pins_processing', exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        '/mnt/HDD/training_checkpoints/KerasCatsVsDogsExperiments/pins_processing/{epoch:02d}.h5',
        verbose=1,
        # save_best_only=True,
        # monitor="mAP",
        # mode='max'
    )



    model.fit_generator(
        train_generator,
        steps_per_epoch=1000,  # nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=(nb_validation_samples // batch_size) or 1)

    model.save_weights('train_session_1.h5')


if __name__ == '__main__':
    train()
