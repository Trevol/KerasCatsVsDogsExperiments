from utils.TrueLabelsReader import TrueLabelsReader
from utils.VideoPlayback import VideoPlayback, KbdKeys
from utils.VideoClassificationCsvLogReader import VideoClassificationCsvLogReader
import cv2
import numpy as np


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
            print(misclassifiedLen)
            if misclassifiedLen == 0:
                continue
            video = VideoPlayback(videoPath)
            # TODO: warn if misclassified frames contains ones from train directory (!!!model misclassified training frames)
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
        elif key == ord('s'):
            cls._moveMisclassifiedFramesToTmpDir(videoId, misclassifiedFrames, trueLog)
        elif key == '':
            # move frame to val
            pass

        return key, currentIndex

    @classmethod
    def _moveMisclassifiedFramesToTmpDir(cls, videoId, misclassifiedFrames, trueLog):
        # TODO: move misclassified frames to [dataset/tmp/{videoId}/busy, dataset/tmp/{videoId}/clear]
        # TODO: DON'T MOVE from validation directory!!!
        print('Move!!!!')

    def run(self):
        s = '5_12'
        map = [
            # ('D:/DiskE/Computer_Vision_Task/video_6.mp4', 6, f'classificationLogs/{s}_video_6_classified.csv', []),
            ('D:/DiskE/Computer_Vision_Task/video_2.mp4', 2, f'classificationLogs/{s}_video_2_classified.csv', [])
        ]
        trueLog = TrueLabelsReader('dataset', ClassificationMeta)

        self.calcMisclassifiedFrames(map, trueLog)
        self.showMisclassifiedFrames(map, trueLog)


if __name__ == '__main__':
    App_().run()
