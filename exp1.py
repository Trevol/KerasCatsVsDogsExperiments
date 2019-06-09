from dataset import makeDataset

train_gen, val_gen = makeDataset('data', (150, 150), 1)
# item = next(train_gen)
# data, labels = item
print(train_gen.labels)
print(train_gen.class_indices)
