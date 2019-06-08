from exp_2_train import model, img_width, img_height
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image


def imageToBatch(imagePath):
    img = load_img(imagePath, target_size=(img_width, img_height))  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x / 255.  # rescale to [0, 1]
    return x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)


cat = imageToBatch('data/train/cats/cat.10.jpg')
dog = imageToBatch('data/train/dogs/dog.10.jpg')

print('no weights. cat. predict', model.predict(cat))
print('no weights. cat. predict_classes', model.predict_classes(cat))
print('no weights. dog. predict', model.predict(dog))
print('no weights. dog. predict_classes', model.predict_classes(dog))

model.load_weights('first_try.h5')

print('With weights. cat. predict', model.predict(cat))
print('With weights. cat. predict_classes', model.predict_classes(cat))
print('With weights. dog. predict', model.predict(dog))
print('With weights. dog. predict_classes', model.predict_classes(dog))
