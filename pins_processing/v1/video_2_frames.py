import cv2


def video_frames(cap, startFrom=None):
    if startFrom:
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrom)
    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        yield pos, frame
    cap.release()


def main():
    map = [
        ('D:/DiskE/Computer_Vision_Task/video_6.mp4', 'dataset/all_frames/video_6/all', '_6'),
        ('D:/DiskE/Computer_Vision_Task/video_2.mp4', 'dataset/all_frames/video_2/all', '_2')
    ]
    params = (cv2.IMWRITE_JPEG_QUALITY, 100)
    for videoPath, framesDir, suffix in map:
        cap = cv2.VideoCapture(videoPath)
        for pos, frame in video_frames(cap):
            framePath = framesDir + f'/{pos:04d}{suffix}.jpg'
            cv2.imwrite(framePath, frame, params)
            if pos > 0 and pos % 500 == 0:
                print(f'{pos} processed')

        cap.release()


if __name__ == '__main__':
    main()
