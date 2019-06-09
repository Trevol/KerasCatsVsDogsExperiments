from keras.preprocessing.image import load_img, img_to_array
from model import makeModel
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw


def imageToBatch(imagePath, targetSize):
    img = load_img(imagePath, target_size=targetSize)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array
    x = x / 255.  # rescale to [0, 1]
    return x, x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, *targetSize)


def load_n_predict(path):
    pass

def main():
    classIndices = {'in_process': 0, 'stable': 1}
    size = (256, 256)
    model = makeModel(size, weights='train_session_1.h5')

    dir = r'D:\DiskE\Computer_Vision_Task\frames_2'
    files = [
        'f2_0456_30400.00_30.40.jpg',
        'f2_0751_50066.67_50.07.jpg'
    ]

    for file in files:
        path = os.path.join(dir, file)
        img, batch = imageToBatch(path, size)

        class_ = model.predict_classes(batch)[0]
        proba = model.predict(batch)[0]

        f, ax = plt.subplots(1, 1)
        ax.set_title(f'{class_}  {proba}')
        ax.imshow(img)
        plt.show()


main()

# cat = imageToBatch('all_data/validation/cats/cat.10000.jpg', inputSize)
# dog = imageToBatch('all_data/validation/dogs/dog.10000.jpg', inputSize)
#
# print('no weights. cat. predict', model.predict(cat))
# print('no weights. cat. predict_classes', model.predict_classes(cat))
# print('no weights. dog. predict', model.predict(dog))
# print('no weights. dog. predict_classes', model.predict_classes(dog))
#
# model.load_weights('first_try.h5')
#
# print('With weights. cat. predict', model.predict(cat))
# print('With weights. cat. predict_classes', model.predict_classes(cat))
# print('With weights. dog. predict', model.predict(dog))
# print('With weights. dog. predict_classes', model.predict_classes(dog))
