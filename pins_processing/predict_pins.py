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


def video_frames(cap, targetSize, startFrom=None):
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


def videoWriter(videoCapture: cv2.VideoCapture, videoPath):
    cc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' ('M', 'J', 'P', 'G')
    # videoOut = cv2.VideoWriter('/mnt/HDD/Rec_15_720_out_76.mp4', fourcc, videoIn.fps(), videoIn.resolution())
    w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    return cv2.VideoWriter(videoPath, cc, videoCapture.get(cv2.CAP_PROP_FPS), size)


def classifyVideo(model, inputSize, srcVideoPath, classifiedVideoPath):
    classIndices = {'in_process': 0, 'stable': 1}
    indexClasses = {0: 'in_process', 1: 'stable'}
    colors = {0: (0, 0, 255), 1: (0, 220, 0)}

    cap = cv2.VideoCapture(srcVideoPath)
    writer = videoWriter(cap, classifiedVideoPath)
    # writer = None

    for pos, bgrImg, batch in video_frames(cap, inputSize, startFrom=0):
        class_ = model.predict_classes(batch)[0, 0]
        proba = model.predict(batch)[0, 0]

        color = colors[class_]
        cv2.putText(bgrImg, f'{pos}', (15, 22), cv2.FONT_HERSHEY_COMPLEX, 1, color)
        info = f'{class_} {indexClasses[class_]}  {proba:.5f}'
        cv2.putText(bgrImg, info, (15, 52), cv2.FONT_HERSHEY_COMPLEX, 1, color)

        if writer: writer.write(bgrImg)

        if pos % 500 == 0:
            print('Processed: ', pos)

        # cv2.imshow('Image', bgrImg)
        # if cv2.waitKey(1) in (27,):
        #     break

    cap.release()
    if writer: writer.release()


def main():
    size = (256, 256)

    weights = '4_50_0.1156_0.9848_0.6352_0.9524.h5'
    model = makeModel(size, weights='weights/' + weights)

    videoMap = [
        (r'D:\DiskE\Computer_Vision_Task\video_6.mp4', 'classificationLogs/video_6_classified_4_50.csv'),
        (r'D:\DiskE\Computer_Vision_Task\video_2.mp4', 'classificationLogs/video_2_classified_4_50.csv')
    ]

    for srcVideoPath, classifiedVideoPath in videoMap:
        logClassification(model, size, srcVideoPath, classifiedVideoPath)


main()
