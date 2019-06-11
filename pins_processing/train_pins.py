from model import makeModel
from dataset import makeDataset
import keras
import os


def train():
    inputSize = (256, 256)
    startWithWeights = '/mnt/HDD/training_checkpoints/KerasCatsVsDogsExperiments/pins_processing/2/11_0.1113_0.9576_1.0759_0.9231.h5'
    model = makeModel(inputSize, compileForTraining=True, weights=startWithWeights)

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
    train_generator, validation_generator = makeDataset('dataset', inputSize, batch_size, train_augmentations)

    nb_train_samples = train_generator.samples
    nb_validation_samples = validation_generator.samples

    checkpointDir = '/mnt/HDD/training_checkpoints/KerasCatsVsDogsExperiments/pins_processing/3'
    os.makedirs(checkpointDir, exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(checkpointDir, '{epoch:02d}_{loss:.4f}_{acc:.4f}_{val_loss:.4f}_{val_acc:.4f}.h5'),
        verbose=1,
        # save_best_only=True,
        # monitor="mAP",
        # mode='max'
    )

    validation_steps = nb_validation_samples // batch_size
    if validation_steps < 2:
        validation_steps = 2
    model.fit_generator(
        train_generator,
        steps_per_epoch=1000,  # nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=validation_steps)


if __name__ == '__main__':
    train()
