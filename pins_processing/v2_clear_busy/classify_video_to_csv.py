from model import makeModel
import os
import cv2
import time


def toBatch(image, targetSize):
    x = cv2.resize(image, targetSize, interpolation=cv2.INTER_NEAREST) / 255.
    return x.reshape((1,) + x.shape)


# def static_files(targetSize):
#     dir = r'D:\DiskE\Computer_Vision_Task\frames_2'
#     files = [
#         'f2_0456_30400.00_30.40.jpg',
#         'f2_0751_50066.67_50.07.jpg'
#     ]
#     for file in files:
#         path = os.path.join(dir, file)
#         bgr = cv2.imread(path)
#         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#         batch = toBatch(rgb, targetSize)
#         yield bgr, batch


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


class VideoClassificationCsvLogWriter:
    def __init__(self, logFilePath):
        self.logFilePath = logFilePath
        self._file = None
        self._writesCount = 0

    def write(self, frameNum, label, proba):
        if self._file is None:
            self._file = open(self.logFilePath, mode='w')

        if self._writesCount > 0:
            self._file.write('\n')
        row = f'{frameNum},{label},{proba}'
        self._file.write(row)
        self._writesCount += 1

    def release(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def logClassification(model, inputSize, srcVideoPath, logFilePath):
    cap = cv2.VideoCapture(srcVideoPath)
    log = VideoClassificationCsvLogWriter(logFilePath)
    baseVideoName = os.path.basename(srcVideoPath)

    t0 = time.time()
    for pos, bgrImg, batch in video_frames(cap, inputSize, startFrom=0):
        class_, proba = model.predict_classes_probabilities(batch)
        class_ = class_[0, 0]
        proba = proba[0, 0]
        log.write(pos, class_, proba)
        if pos > 0 and pos % 500 == 0:
            print(f'{baseVideoName}. Processed: {pos}. Time elapsed: {time.time() - t0:.1f} s')

    log.release()
    cap.release()


def main():
    size = (256, 256)

    weights = '4_26_1.1794_0.9155_0.0883_0.9804.h5'
    model = makeModel(size, weights='weights/' + weights)

    epoch = '4_26'
    videoMap = [
        (r'D:\DiskE\Computer_Vision_Task\video_6.mp4', f'classificationLogs/{epoch}_video_6_classified.csv'),
        # (r'D:\DiskE\Computer_Vision_Task\video_2.mp4', f'classificationLogs/{epoch}_video_2_classified.csv')
    ]

    for srcVideoPath, classifiedVideoPath in videoMap:
        logClassification(model, size, srcVideoPath, classifiedVideoPath)


main()
