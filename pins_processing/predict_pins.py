from keras.preprocessing.image import load_img, img_to_array
from model import makeModel
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw, Image
import cv2


def toBatch(image, targetSize):
    x = cv2.resize(image, targetSize, interpolation=cv2.INTER_NEAREST) / 255.
    return x.reshape((1,) + x.shape)


def static_files(targetSize):
    dir = r'D:\DiskE\Computer_Vision_Task\frames_2'
    files = [
        'f2_0456_30400.00_30.40.jpg',
        'f2_0751_50066.67_50.07.jpg'
    ]
    for file in files:
        path = os.path.join(dir, file)
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        batch = toBatch(rgb, targetSize)
        yield bgr, batch


def video_frames(targetSize, startFrom=None):
    path = r'D:\DiskE\Computer_Vision_Task\video_2.mp4'
    cap = cv2.VideoCapture(path)
    if startFrom:
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrom)
    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield pos, frame, toBatch(rgb, targetSize)
    cap.release()


def main():
    classIndices = {'in_process': 0, 'stable': 1}
    indexClasses = {0: 'in_process', 1: 'stable'}
    size = (256, 256)
    model = makeModel(size, weights='50_0.1931_0.9345_2.7987_0.8333.h5')

    for pos, bgrImg, batch in video_frames(size, startFrom=600):
        class_ = model.predict_classes(batch)[0, 0]
        proba = model.predict(batch)[0, 0]

        cv2.putText(bgrImg, f'{pos}', (15, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        info = f'{class_} {indexClasses[class_]}  {proba:.5f}'
        cv2.putText(bgrImg, info, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        cv2.imshow('Image', bgrImg)
        if cv2.waitKey() in (-1, 27):
            break


main()

# def imageToBatch(imagePath, targetSize):
#     img = load_img(imagePath, target_size=None)  # this is a PIL image
#     img = img.resize(targetSize, Image.NEAREST)
#     x = img_to_array(img)  # this is a Numpy array
#     x = x / 255.  # rescale to [0, 1]
#     return x, x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, *targetSize, 3)
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
