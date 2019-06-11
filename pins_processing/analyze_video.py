import cv2


class VideoPlayback:
    def __init__(self, videoPath):
        self.cap = cv2.VideoCapture(videoPath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __currentPos(self):
        assert self.cap
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def setPos(self, pos):
        assert self.cap
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def backward(self, numFrames=1):
        assert self.cap
        newPos = self.__currentPos() - numFrames - 1
        if newPos < 0:
            newPos = 0
        self.setPos(newPos)

    def frames(self):
        if not self.cap:
            raise Exception('VideoCapture already released')
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield self.__currentPos() - 1, frame


L_ARROW = 2424832


# check
# 402 (req: in_process), [1556-1565] (req: in_process), 1614 (req: in_process)
# 2547 - in_proc
def main():
    path = 'videos/video_6_classified_4_17.mp4'
    with VideoPlayback(path) as video:
        video.setPos(0)
        for pos, frame in video.frames():
            cv2.imshow('Video', frame)
            key = cv2.waitKeyEx()
            if key in (27,):
                break
            elif key == L_ARROW:
                video.backward()


if __name__ == '__main__':
    main()
