from utils.VideoPlayback import VideoPlayback, KbdKeys
from utils.VideoClassificationCsvLogReader import VideoClassificationCsvLogReader
import os
import cv2


class TrueLabelsReader:
    def __init__(self, trueClassificationDir, meta):
        self._framesClasses = {}

        for className in meta.NameToId:
            for fileName in os.listdir(os.path.join(trueClassificationDir, className)):
                fileName = os.path.splitext(fileName)[0]  # 0034_6.jpg => 0034_6
                framePos = int(fileName.split('_')[0])  # 0036_6 => 0036 => 36
                self._framesClasses[framePos] = meta.NameToId[className]

    def byFramePos(self, framePos):
        return self._framesClasses[framePos]


class ClassificationMeta:
    IdToName = {0: 'busy', 1: 'clear'}
    NameToId = {v: k for k, v in IdToName.items()}
    Colors = {0: (0, 0, 255), 1: (0, 220, 0)}


def putInfo(frame, pos, trueClass, computedClass, computedProba):
    idToName = ClassificationMeta.IdToName
    colors = ClassificationMeta.Colors

    color = colors[trueClass]
    cv2.putText(frame, f'{pos}', (15, 22), cv2.FONT_HERSHEY_COMPLEX, 1, color)
    info = f'TRUE: {trueClass} {idToName[trueClass]}'
    cv2.putText(frame, info, (15, 52), cv2.FONT_HERSHEY_COMPLEX, 1, color)

    color = colors[computedClass]
    info = f'COMPUTED: {computedClass} {idToName[computedClass]}  {computedProba:.5f}'
    cv2.putText(frame, info, (15, 82), cv2.FONT_HERSHEY_COMPLEX, 1, color)


def main():
    map = [
        ('D:/DiskE/Computer_Vision_Task/video_6.mp4', 'classificationLogs/video_6_classified_1_10.csv',
         'dataset/video_6', []),
        ('D:/DiskE/Computer_Vision_Task/video_2.mp4', 'classificationLogs/video_2_classified_1_10.csv',
         'dataset/video_2', [])
    ]

    for i, (videoPath, classificationLogPath, trueClassificationDir, misclassifiedFrames) in enumerate(map):
        video = VideoPlayback(videoPath)
        computedLog = VideoClassificationCsvLogReader(classificationLogPath)
        trueLog = TrueLabelsReader(trueClassificationDir, ClassificationMeta)
        for pos in range(video.framesCount()):
            computedClass, computedProba = computedLog.byFramePos(pos)
            trueClass = trueLog.byFramePos(pos)
            if computedClass != trueClass:
                misclassifiedFrames.append([pos, trueClass, computedClass, computedProba])
        video.release()

    for videoPath, _, _, misclassifiedFrames in map:
        print(len(misclassifiedFrames))
        video = VideoPlayback(videoPath)
        for pos, trueClass, computedClass, computedProba in misclassifiedFrames:
            video.setPos(pos)
            frame = video.readFrame()
            putInfo(frame, pos, trueClass, computedClass, computedProba)
            cv2.imshow('Video', frame)
            if cv2.waitKeyEx() == KbdKeys.ESC:
                break
        video.release()


if __name__ == '__main__':
    main()
