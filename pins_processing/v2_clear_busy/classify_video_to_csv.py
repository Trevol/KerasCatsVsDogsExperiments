import numpy as np
from model import makeModel
import os
import cv2
import time


class FramesBatcher:
    def __init__(self, videoCapture, batchSize, targetImgSize):
        assert batchSize > 0
        self.videoCapture = videoCapture
        self.batchSize = batchSize
        self.targetImgSize = targetImgSize

    @staticmethod
    def _getFramePos(videoCapture):
        return int(videoCapture.get(cv2.CAP_PROP_POS_FRAMES))

    @staticmethod
    def batch(frames, targetSize):
        if len(frames) == 0:
            return None
        preparedFrames = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, targetSize, interpolation=cv2.INTER_NEAREST) / 255.
            preparedFrames.append(frame)
        return np.stack(preparedFrames)

    def nextBatch(self):
        # accumulate batches
        frames = []
        framePositions = []

        for _ in range(self.batchSize):
            pos = self._getFramePos(self.videoCapture)
            ret, frame = self.videoCapture.read()
            if not ret:
                break
            framePositions.append(pos)
            frames.append(frame)

        batch = self.batch(frames, self.targetImgSize)
        return framePositions, frames, batch

    def batches(self):
        while True:
            framePositions, frames, batch = self.nextBatch()
            if len(framePositions) == 0:
                break
            yield framePositions, frames, batch


def video_frames_batched(videoCapture, targetSize, batchSize, startFrom=None):
    if startFrom:
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, startFrom)
    batcher = FramesBatcher(videoCapture, batchSize, targetSize)
    return batcher.batches()


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


def classifyAndLog(model, inputSize, srcVideoPath, logFilePath, batchSize):
    cap = cv2.VideoCapture(srcVideoPath)
    log = VideoClassificationCsvLogWriter(logFilePath)
    baseVideoName = os.path.basename(srcVideoPath)

    pos = -1
    t0 = time.time()
    for framePositions, frames, batch in video_frames_batched(cap, inputSize, batchSize, startFrom=0):
        classes, probabilities = model.predict_classes_probabilities(batch, batchSize)

        for pos, (class_,), (proba,) in zip(framePositions, classes, probabilities):
            log.write(pos, class_, proba)
            if pos > 0 and pos % 1000 == 0:
                print(f'{baseVideoName}. Processed: {pos}. Time elapsed: {time.time() - t0:.1f} s')

    print(f'{baseVideoName}. Processed: {pos}. Time elapsed: {time.time() - t0:.1f} s')

    log.release()
    cap.release()


def lastWeights(weightsDir):
    weightsFiles = [file for file in os.listdir(weightsDir) if os.path.splitext(file)[-1] == '.h5']
    if len(weightsFiles) == 0:
        return None, None

    def epochFn(file):
        p = file.split('_')
        return (int(p[0]), int(p[1]))

    lastFile = sorted(weightsFiles, key=epochFn)[-1]

    parts = lastFile.split('_')
    epoch = f'{parts[0]}_{parts[1]}'
    return epoch, os.path.join(weightsDir, lastFile)


def main():
    size = (256, 256)

    weightsDir = '/HDD_DATA/training_checkpoints/KerasCatsVsDogsExperiments/pins_processing_v2/2'
    epoch, weights = lastWeights(weightsDir)
    print(f'Classifying epoch {epoch}. Weights: {weights}')

    model = makeModel(size, weights=weights)

    videoMap = [
        ('/HDD_DATA/Computer_Vision_Task/Video_2.mp4', f'classificationLogs/{epoch}_video_2_classified.csv'),
        ('/HDD_DATA/Computer_Vision_Task/Video_6.mp4', f'classificationLogs/{epoch}_video_6_classified.csv')
    ]

    framesBatchSize = 64
    for srcVideoPath, classifiedVideoPath in videoMap:
        classifyAndLog(model, size, srcVideoPath, classifiedVideoPath, framesBatchSize)


main()
