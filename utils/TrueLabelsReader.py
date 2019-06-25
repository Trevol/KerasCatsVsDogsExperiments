import os
import itertools


class TrueLabelsReader:
    def __init__(self, trueClassificationDir, meta):
        self._videoFramesClasses = {}
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

        # for className in meta.NameToId:
        #     for fileName in os.listdir(os.path.join(trueClassificationDir, className)):
        #         fileName = os.path.splitext(fileName)[0]  # 0034_6.jpg => 0034_6
        #         framePos = int(fileName.split('_')[0])  # 0036_6 => 0036 => 36
        #         self._framesClasses[framePos] = meta.NameToId[className]

    @staticmethod
    def _classDirectories(searchRootDirs, classNames):
        for searchRotDir in searchRootDirs:
            for root, dirs, files in os.walk(searchRotDir):
                for dir in dirs:
                    if dir in classNames:
                        yield dir, os.path.join(root, dir)

    def byFramePos(self, videoId, framePos):
        return self._videoFramesClasses[(videoId, framePos)]


if __name__ == '__main__':
    def top(n, iterable):
        i = 0
        for item in iterable:
            yield item
            i += 1
            if i >= n:
                break


    def __tttt():
        r = TrueLabelsReader('../pins_processing/v2_clear_busy/dataset/train', Meta)
        print(r.byFramePos(2, 0), 0)


    class Meta:
        NameToId = dict(busy=0, clear=1)


    __tttt()
