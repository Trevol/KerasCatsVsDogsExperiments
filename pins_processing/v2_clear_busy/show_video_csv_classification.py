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
    # 0038_6   0130_6 0252
    # 0060_2
    epoch = '4_11'
    videoMap = [
        ('D:/DiskE/Computer_Vision_Task/video_6.mp4', f'classificationLogs/{epoch}_video_6_classified.csv'),
        ('D:/DiskE/Computer_Vision_Task/video_2.mp4', f'classificationLogs/{epoch}_video_2_classified.csv')
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
