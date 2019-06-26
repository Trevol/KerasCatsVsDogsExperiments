from utils.TrueLabelsReader import TrueLabelsReader
from utils.VideoPlayback import VideoPlayback, KbdKeys
from utils.VideoClassificationCsvLogReader import VideoClassificationCsvLogReader
import cv2
import numpy as np


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
    s = '4_26'
    map = [
        ('D:/DiskE/Computer_Vision_Task/video_6.mp4', 6, f'classificationLogs/{s}_video_6_classified.csv',
         'dataset/video_6', []),
        ('D:/DiskE/Computer_Vision_Task/video_2.mp4', 2, f'classificationLogs/{s}_video_2_classified.csv',
         'dataset/video_2', [])
    ]
    trueLog = TrueLabelsReader('dataset', ClassificationMeta)

    for i, (videoPath, videoId, classificationLogPath, trueClassificationDir, misclassifiedFrames) in enumerate(map):
        video = VideoPlayback(videoPath)
        computedLog = VideoClassificationCsvLogReader(classificationLogPath)

        for pos in range(video.framesCount()):
            computedClass, computedProba = computedLog.byFramePos(pos)
            trueClass = trueLog.byFramePos(videoId, pos)
            if computedClass != trueClass:
                misclassifiedFrames.append([pos, trueClass, computedClass, computedProba])
        video.release()

    for videoPath, videoId, _, _, misclassifiedFrames in map:
        misclassifiedLen = len(misclassifiedFrames)
        print(misclassifiedLen)
        if misclassifiedLen == 0:
            continue
        video = VideoPlayback(videoPath)

        i = 0
        while True:
            pos, trueClass, computedClass, computedProba = misclassifiedFrames[i]
            video.setPos(pos)
            frame = video.readFrame()
            putInfo(frame, pos, trueClass, computedClass, computedProba)
            cv2.imshow('Video', frame)

            key = cv2.waitKeyEx()
            if key == KbdKeys.ESC:
                break
            elif key == KbdKeys.L_ARROW:
                i -= 1
            elif key == KbdKeys.R_ARROW:
                i += 1
            i = np.clip(i, 0, misclassifiedLen - 1)

        video.release()


if __name__ == '__main__':
    main()
