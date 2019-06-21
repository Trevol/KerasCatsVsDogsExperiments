import os
import random
import numpy as np


def chooseItems(srcList, count, std):
    assert count > 1
    srcList = sorted(srcList)
    lastIndex = len(srcList) - 1
    dist = lastIndex / (count - 1)

    indices = [np.clip(int(round(i * dist)) + random.randint(-std, std), 0, lastIndex)
               for i in range(count)]

    result = []
    for i in indices:
        result.append(srcList[i])
    return result


def moveFiles(files, from_, to):
    # https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python
    for file in files:
        src = os.path.join(from_, file)
        dst = os.path.join(to, file)
        os.rename(src, dst)

def restore():
    pass

def main():
    # fromDir toDir cnt stdDeviation
    dirsConfig = [
        ('dataset/video_2/clear', 'dataset/train/clear', 40, 11),
        ('dataset/video_2/busy', 'dataset/train/busy', 40, 11),
        ('dataset/video_2/clear', 'dataset/validation/clear', 20, 15),
        ('dataset/video_2/busy', 'dataset/validation/busy', 20, 15)
    ]
    # TODO: how to choose frames? cluster by SSMI (structured similarity index) or
    # TODO: google "cluster images by similarity"
    for from_, to, filesCount, std in dirsConfig:
        os.makedirs(to, exist_ok=True)
        choosenFiles = chooseItems(os.listdir(from_), filesCount, std)
        moveFiles(choosenFiles, from_, to)
        print(choosenFiles)




if __name__ == '__main__':
    main()
