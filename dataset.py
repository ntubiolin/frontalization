
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder


# class myImageFolder(ImageFolder):
#     @property
#     def train_labels(self):
#         warnings.warn("train_labels has been renamed targets")
#         return self.targets

#     def __init__(self, root, transform=None, target_transform=None):
#         super(myImageFolder, self).__init__(root, transform, target_transform)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        try:
            # this is what ImageFolder normally returns
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
        except:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(self.imgs[index][0])
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple[0], path)
        # print(tuple_with_path)
        return tuple_with_path


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # instantiate the dataset and dataloader
    data_dir = "IJB_A_cropped_imgs"
    transform = transforms.Compose([transforms.CenterCrop(128),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    dataset = ImageFolderWithPaths(data_dir, transform)
    dataloader = DataLoader(dataset)

    # iterate over data
    for inputs, paths in dataloader:
        # use the above variables freely
        print(inputs, paths)
