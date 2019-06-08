from exp_2_train import model, img_width, img_height
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

img = load_img('data/train/cats/cat.0.jpg', target_size=(img_width, img_height))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x / 255.
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

result = model.predict(x)
print(result)

model.load_weights('first_try.h5')
result = model.predict(x)
print(result)
result = model.predict_classes(x)
print(result)


img = load_img('data/train/dogs/dog.0.jpg', target_size=(img_width, img_height))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x / 255.
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

result = model.predict(x)
print(result)
result = model.predict_classes(x)
print(result)
