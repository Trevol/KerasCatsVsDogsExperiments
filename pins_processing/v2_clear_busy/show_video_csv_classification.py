import cv2

from utils.KbdKeys import KbdKeys
from utils.VideoClassificationCsvLogReader import VideoClassificationCsvLogReader
from utils.VideoPlayback import VideoPlayback


def putInfo(frame, framePos, class_, probability):
    # classIndices = {'in_process': 0, 'stable': 1}
    indexClasses = {0: 'busy', 1: 'clear'}
    colors = {0: (0, 0, 255), 1: (0, 220, 0)}

    color = colors[class_]
    cv2.putText(frame, f'{framePos}', (15, 22), cv2.FONT_HERSHEY_COMPLEX, 1, color)
    info = f'{class_} {indexClasses[class_]}  {probability:.5f}'
    cv2.putText(frame, info, (15, 52), cv2.FONT_HERSHEY_COMPLEX, 1, color)


def main():
    videoMap = [
        ('D:/DiskE/Computer_Vision_Task/video_6.mp4', 'classificationLogs/video_6_classified_2_20.csv'),
        # ('D:/DiskE/Computer_Vision_Task/video_2.mp4', 'classificationLogs/video_2_classified_2_20.csv')
    ]

    for videoPath, annotationLog in videoMap:
        logReader = VideoClassificationCsvLogReader(annotationLog)
        video = VideoPlayback(videoPath)
        video.setPos(0)

        for pos, frame in video.frames():
            class_, probability = logReader.byFramePos(pos)
            putInfo(frame, pos, class_, probability)
            cv2.imshow('Video', frame)

            if video.handleKey() == KbdKeys.ESC:
                break

        video.release()


if __name__ == '__main__':
    main()
