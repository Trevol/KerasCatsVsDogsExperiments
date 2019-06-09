from keras.preprocessing.image import ImageDataGenerator
import os


def makeDataset(datasetPath, inputSize, batchSize):
    train_data_dir = os.path.join(datasetPath, 'train')
    validation_data_dir = os.path.join(datasetPath, 'validation')

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=inputSize,
        batch_size=batchSize,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=inputSize,
        batch_size=batchSize,
        class_mode='binary')

    return train_generator, validation_generator
