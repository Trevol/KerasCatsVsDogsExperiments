import os


def main():
    # fromDir toDir cnt
    dirsConfig = [
        ('dataset/video_2/clear', 'dataset/train/clear', 20),
        ('dataset/video_2/busy', 'dataset/train/busy', 20),
        ('dataset/video_2/clear', 'dataset/validation/clear', 20),
        ('dataset/video_2/busy', 'dataset/validation/busy', 20)
    ]
    for from_, to, count in dirsConfig:
        os.makedirs(to, exist_ok=True)


if __name__ == '__main__':
    main()
