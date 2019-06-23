from model import makeModel
from dataset import makeDataset
import keras
import os


def train():
    inputSize = (256, 256)
    trainSessionId = 3
    baseCheckpointsDir = '/mnt/HDD/training_checkpoints/KerasCatsVsDogsExperiments/pins_processing_v2'

    startWithWeights = f'{baseCheckpointsDir}/2/2_20_1.9109_0.8782_0.1993_0.9875.h5'
    # startWithWeights = None

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

    checkpointDir = f'{baseCheckpointsDir}/{trainSessionId}'
    os.makedirs(checkpointDir, exist_ok=True)

    checkpointNameTemplate = str(trainSessionId) + '_{epoch:02d}_{loss:.4f}_{acc:.4f}_{val_loss:.4f}_{val_acc:.4f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(checkpointDir, checkpointNameTemplate),
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
