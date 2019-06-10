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


def error_frames_video_2():
    f = [
        809,
        1197,  # 1199
        1202,
        1557,  # 1558
        1560,  # 1565
        1614,
    ]


def main():
    classIndices = {'in_process': 0, 'stable': 1}
    indexClasses = {0: 'in_process', 1: 'stable'}
    colors = {0: (0, 0, 255), 1: (0, 220, 0)}

    size = (256, 256)

    # weights = '11_0.1113_0.9576_1.0759_0.9231.h5'    # 809-0.81523  810-0.99917 811-0.10084
    weights = '03_0.1970_0.9487_0.3971_0.8462.h5'    #   809-0.99674  810-0.99912 811-0.98278
    model = makeModel(size, weights=weights)

    for pos, bgrImg, batch in video_frames(size, startFrom=1557):
        class_ = model.predict_classes(batch)[0, 0]
        proba = model.predict(batch)[0, 0]

        color = colors[class_]
        cv2.putText(bgrImg, f'{pos}', (15, 22), cv2.FONT_HERSHEY_COMPLEX, 1, color)
        info = f'{class_} {indexClasses[class_]}  {proba:.5f}'
        cv2.putText(bgrImg, info, (15, 52), cv2.FONT_HERSHEY_COMPLEX, 1, color)

        cv2.imshow('Image', bgrImg)
        if cv2.waitKey() in (-1, 27):
            break


main()
