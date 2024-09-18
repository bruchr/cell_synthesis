from torch.utils.data import DataLoader

from dataset import CycleGANDataset
from models.cyclegan import CycleGANModel
from options.options import options_parser


opt = options_parser()  # Pass input options
# Create dataset, dataloader and model
dataset = CycleGANDataset(opt)
dataloader = DataLoader(dataset, batch_size=opt["batch_size"] if opt["mode"]=='train' else 1, shuffle=opt["mode"]=='train', num_workers=0)
model = CycleGANModel(opt)


if opt["mode"] == "train":
    model.train(dataloader)
if opt["mode"] == "inference":
    model.inference(dataloader)
print(f"Finished {opt['mode']}")
