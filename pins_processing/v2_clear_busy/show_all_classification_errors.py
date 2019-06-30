from utils.TrueLabelsReader import TrueLabelsReader
from utils.VideoPlayback import VideoPlayback, KbdKeys
from utils.VideoClassificationCsvLogReader import VideoClassificationCsvLogReader
import cv2
import numpy as np
from warnings import warn
import os


class ClassificationMeta:
    IdToName = {0: 'busy', 1: 'clear'}
    NameToId = {v: k for k, v in IdToName.items()}
    Colors = {0: (0, 0, 255), 1: (0, 220, 0)}


def putInfo(frame, pos, trueClass, computedClass, computedProba, trueLabelDir):
    idToName = ClassificationMeta.IdToName
    colors = ClassificationMeta.Colors

    color = colors[trueClass]
    cv2.putText(frame, f'{pos}', (15, 22), cv2.FONT_HERSHEY_COMPLEX, 1, color)
    info = f'TRUE: {trueClass} {idToName[trueClass]}  {trueLabelDir}'
    cv2.putText(frame, info, (15, 52), cv2.FONT_HERSHEY_COMPLEX, 1, color)

    color = colors[computedClass]
    info = f'COMPUTED: {computedClass} {idToName[computedClass]}  {computedProba:.5f}'
    cv2.putText(frame, info, (15, 82), cv2.FONT_HERSHEY_COMPLEX, 1, color)


class App_:
    @staticmethod
    def calcMisclassifiedFrames(map, trueLog):
        for videoPath, videoId, classificationLogPath, misclassifiedFrames in map:
            video = VideoPlayback(videoPath)
            computedLog = VideoClassificationCsvLogReader(classificationLogPath)
            lastFramePos = trueLog.lastFramePos(videoId)
            for pos in range(lastFramePos or video.framesCount()):
                computedClass, computedProba = computedLog.byFramePos(pos)
                trueClass = trueLog.classByFramePos(videoId, pos)
                if computedClass != trueClass:
                    misclassifiedFrames.append([pos, trueClass, computedClass, computedProba])
            video.release()

    @classmethod
    def showMisclassifiedFrames(cls, map, trueLog):
        for videoPath, videoId, _, misclassifiedFrames in map:
            misclassifiedLen = len(misclassifiedFrames)

            from itertools import groupby
            statsByTrueLabel = [ (k, len(v)) for k, v in groupby(misclassifiedLen, key=lambda i: i[1])]
            print(misclassifiedLen, statsByTrueLabel)

            return

            if misclassifiedLen == 0:
                continue

            video = VideoPlayback(videoPath)

            # warn if misclassified frames contains ones from train directory (!!!model misclassified training frames)
            for pos, trueClass, computedClass, computedProba in misclassifiedFrames:
                trueLabelDir = trueLog.dirByFramePos(videoId, pos)
                if 'train' in trueLabelDir:
                    warn(f'train frame {pos:04d}_{videoId} is misclassified!!!')

            currentIndex = 0
            while True:
                pos, trueClass, computedClass, computedProba = misclassifiedFrames[currentIndex]
                frame = video.readFrame(pos)
                trueLabelDir = trueLog.dirByFramePos(videoId, pos)
                putInfo(frame, pos, trueClass, computedClass, computedProba, trueLabelDir)
                cv2.imshow('Video', frame)

                actionContext = [currentIndex, 0, misclassifiedLen - 1, frame, misclassifiedFrames, trueLog, videoId]
                key, currentIndex = cls.handleAction(actionContext)

                if key == KbdKeys.ESC:
                    break

            video.release()

    @classmethod
    def handleAction(cls, actionContext):
        currentIndex, minIndex, maxIndex, frame, misclassifiedFrames, trueLog, videoId = actionContext
        key = cv2.waitKeyEx()
        if key == KbdKeys.L_ARROW:
            currentIndex -= 1
            currentIndex = np.clip(currentIndex, minIndex, maxIndex)
        elif key == KbdKeys.R_ARROW:
            currentIndex += 1
            currentIndex = np.clip(currentIndex, minIndex, maxIndex)
        elif key == ord('S'):
            cls._moveMisclassifiedFramesToTmpDir(videoId, misclassifiedFrames, trueLog)

        return key, currentIndex

    @classmethod
    def _moveMisclassifiedFramesToTmpDir(cls, videoId, misclassifiedFrames, trueLog):
        # move misclassified frames to [dataset/tmp/{videoId}/busy, dataset/tmp/{videoId}/clear]

        # raise if misclassified frames contains ones from train directory (!!!model misclassified training frames)
        for pos, trueClass, computedClass, computedProba in misclassifiedFrames:
            trueLabelDir = trueLog.dirByFramePos(videoId, pos)
            if 'train' in trueLabelDir:
                raise Exception(f'train frame {pos:04d}_{videoId} is misclassified!!!')

        tmpMisclassifiedDir = 'dataset/tmpMisclassified'

        if len(misclassifiedFrames):
            labelNames = ClassificationMeta.NameToId
            for name in labelNames:
                os.makedirs(f'{tmpMisclassifiedDir}/{videoId}/{name}', exist_ok=True)

        for pos, trueClass, computedClass, computedProba in misclassifiedFrames:
            trueLabelDir = trueLog.dirByFramePos(videoId, pos)
            if 'validation' in trueLabelDir:  # DON'T MOVE from validation directory!!!
                continue
            jpeg = f'{pos:04d}_{videoId}.jpg'
            trueLabelFile = os.path.join(trueLabelDir, jpeg)
            if not os.path.isfile(trueLabelFile):
                continue
            trueClassName = ClassificationMeta.IdToName[trueClass]
            newLocation = f'{tmpMisclassifiedDir}/{videoId}/{trueClassName}/{jpeg}'
            os.rename(trueLabelFile, newLocation)

        print('Move!!!!')

    def run(self):
        s = '7_16'
        map = [
            # ('D:/DiskE/Computer_Vision_Task/video_6.mp4', 6, f'classificationLogs/{s}_video_6_classified.csv', []),
            ('D:/DiskE/Computer_Vision_Task/video_2.mp4', 2, f'classificationLogs/{s}_video_2_classified.csv', [])
        ]
        trueLog = TrueLabelsReader('dataset', ClassificationMeta)

        self.calcMisclassifiedFrames(map, trueLog)
        self.showMisclassifiedFrames(map, trueLog)


if __name__ == '__main__':
    mmm = [
        (1, 'busy'),
        (1, 'busy'),
        (2, 'clear'),
        (2, 'clear'),
        (3, 'clear')
    ]
    from itertools import groupby
    groupby(mmm, key=lambda i: i[1])
    # App_().run()
