import csv


class VideoClassificationCsvLogReader:
    def __init__(self, logFilePath):
        self._data = {}
        with open(logFilePath, mode='r', newline='') as file:
            for row in csv.reader(file, delimiter=','):
                framePos = int(row[0])
                class_ = int(row[1])
                probability = float(row[2])
                self._data[framePos] = (class_, probability)

    def byFramePos(self, framePos):
        return self._data[framePos]