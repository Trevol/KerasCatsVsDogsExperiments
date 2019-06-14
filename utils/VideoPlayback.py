import cv2

from utils.KbdKeys import KbdKeys


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

    def handleKey(self):
        key = cv2.waitKeyEx()
        if key == KbdKeys.L_ARROW:
            self.backward()
        return key