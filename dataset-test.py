from dataset import Images
from torch.utils.data import DataLoader

batch_size = 10
ds_dir = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/resized_128/"
images = Images(ds_dir, 'TRAINING.csv', True)
train = DataLoader(images, batch_size=batch_size, num_workers=1, shuffle=True)

print("Training on " + str(len(train)*batch_size) + " images.")

