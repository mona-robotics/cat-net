from torch.utils.data import DataLoader, ConcatDataset

from cat_net import config
from cat_net.models import CATModel
from cat_net.datasets import nordland
from cat_net import experiment

### CONFIGURATION ###
### (defaults in __init__.py) ###
config.data_dir = '/scratch/gridseth/data'
config.results_dir = '/scratch/gridseth/results'

config.experiment_name = 'nordland-small-test'

config.use_cuda = False
config.down_levels = 7
config.innermost_kernel_size = (3, 4)
config.batch_size = 16
config.train_epochs = 5

config.image_load_size = (240, 320) # (180, 320)  # H, W
config.image_final_size = (192, 256) # (144, 256)  # H, W
config.random_crop = True  # if True, crops load_size to final_size, else scales

self.visualize = False

print(config)
config.save_txt()

### INITIALIZE MODEL ###
model = CATModel()

# ### TRAIN AND VALIDATE ###
train_canonical = 'train/fall'
train_data = ConcatDataset(
    [nordland.TorchDataset('train/fall', train_canonical),
     nordland.TorchDataset('train/winter', train_canonical)])
     # nordland.TorchDataset('train/summer', train_canonical),
     # nordland.TorchDataset('train/spring', train_canonical)])

val_canonical = 'validation/fall'
val_data = ConcatDataset(
	[nordland.TorchDataset('validation/fall', val_canonical),
	 nordland.TorchDataset('validation/winter', val_canonical)])
	# nordland.TorchDataset('validation/summer', val_canonical),
	# nordland.TorchDataset('validation/spring', val_canonical)]

experiment.train(model, train_data, val_data)


### TEST ###
test_seqs = ['test/fall', 'test/winter'] #, 'test/summer', 'test/spring']
test_canonical = 'test/fall'

for seq in test_seqs:
    test_data = nordland.TorchDataset(seq, test_canonical)
    experiment.test(model, test_data, label=seq)
