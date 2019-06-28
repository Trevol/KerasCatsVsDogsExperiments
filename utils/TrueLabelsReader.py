import os
import itertools


class TrueLabelsReader:
    def __init__(self, trueClassificationDir, meta):
        self._videoFramesClasses = {}
        self._videoFramesDirs = {}
        classNames = list(meta.NameToId)
        if not isinstance(trueClassificationDir, list):
            trueClassificationDir = [trueClassificationDir]

        for className, dirPath in self._classDirectories(trueClassificationDir, classNames):
            for fileName in os.listdir(dirPath):
                fileName, ext = os.path.splitext(fileName)  # 0034_6.jpg => 0034_6
                if not ext:
                    continue

                split = fileName.split('_')
                framePos, videoId = int(split[0]), int(split[1])  # 0036_6 => 0036 => 36
                self._videoFramesClasses[(videoId, framePos)] = meta.NameToId[className]
                self._videoFramesDirs[(videoId, framePos)] = dirPath

    @staticmethod
    def _classDirectories(searchRootDirs, classNames):
        for searchRotDir in searchRootDirs:
            for root, dirs, files in os.walk(searchRotDir):
                for dir in dirs:
                    if dir in classNames:
                        yield dir, os.path.join(root, dir)

    def classByFramePos(self, videoId, framePos):
        return self._videoFramesClasses[(videoId, framePos)]

    def dirByFramePos(self, videoId, framePos):
        return self._videoFramesDirs[(videoId, framePos)]

    def lastFramePos(self, videoId):
        videoFramesPositions = [framePos for videoId_, framePos in self._videoFramesClasses if videoId_ == videoId]
        return max(videoFramesPositions)


if __name__ == '__main__':
    def __test():
        class Meta:
            NameToId = dict(busy=0, clear=1)

        r = TrueLabelsReader('../pins_processing/v2_clear_busy/dataset/train', Meta)
        print(r.classByFramePos(2, 0), 0)


    __test()
