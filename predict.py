from keras.preprocessing.image import load_img, img_to_array
from model import makeModel


def imageToBatch(imagePath, targetSize):
    img = load_img(imagePath, target_size=targetSize)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array
    x = x / 255.  # rescale to [0, 1]
    return x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, *targetSize)


inputSize = (150, 150)
model = makeModel(inputSize)
cat = imageToBatch('all_data/validation/cats/cat.10000.jpg', inputSize)
dog = imageToBatch('all_data/validation/dogs/dog.10000.jpg', inputSize)

print('no weights. cat. predict', model.predict(cat))
print('no weights. cat. predict_classes', model.predict_classes(cat))
print('no weights. dog. predict', model.predict(dog))
print('no weights. dog. predict_classes', model.predict_classes(dog))

model.load_weights('first_try.h5')

print('With weights. cat. predict', model.predict(cat))
print('With weights. cat. predict_classes', model.predict_classes(cat))
print('With weights. dog. predict', model.predict(dog))
print('With weights. dog. predict_classes', model.predict_classes(dog))
