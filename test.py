import os
import torch
import os.path as op
import torchvision.utils as vutils
# from torch.autograd import Variable
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
from tqdm import tqdm

# from data import ImagePipeline
from dataset import ImageFolderWithPaths
from torch.utils.data import DataLoader
from torchvision import transforms
device = 'cuda'

datapath = 'IJB_A_cropped_imgs'
transform = transforms.Compose([transforms.CenterCrop(128),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

# Generate frontal images from the test set
def frontalize(model, datapath, batch_size, dst_dir):
    dataset = ImageFolderWithPaths(datapath, transform)
    test_pipe_loader = DataLoader(dataset, batch_size=8,
                                  shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch_num, data in tqdm(enumerate(test_pipe_loader),
                                    total=len(test_pipe_loader)):
            profiles, paths = data
            profiles = profiles.to(device)
            generated = model(profiles)
            for i, img in enumerate(generated):
                path = paths[i]
                os.makedirs(op.join(dst_dir, *path.split('/')[1:-1]),
                            exist_ok=True)
                path = op.join(dst_dir, *path.split('/')[1:])
                vutils.save_image(img, path, padding=0, normalize=True)
            vutils.save_image(torch.cat((profiles, generated)),
                              f'output/{batch_num}.jpg',
                              nrow=batch_size, padding=2, normalize=True)
    return


# Load a pre-trained Pytorch model
saved_model = torch.load("pretrained/generator_v0.pt")

frontalize(saved_model, datapath, 8, 'IJB_A_Frontal')

