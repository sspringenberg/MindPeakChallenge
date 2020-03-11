import os
from PIL import Image
from torch.utils.data import Dataset

class MPChallengeDataset(Dataset):
    """
    costruct the MindPeak MindPeak Challenge Dataset
    """

    def __init__(self, data_root, transform=None):

        self.samples = []
        self.transform = transform
        self.class_dict = {"class1": 0, "class2": 1, "class3": 2, "class4": 3}

        for cl in os.listdir(data_root):
            cl_folder = os.path.join(data_root, cl)

            for curr_image in os.listdir(cl_folder):
                curr_image_fpath = os.path.join(cl_folder, curr_image)

                self.samples.append((curr_image_fpath, self.class_dict[cl]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        get returns PIL image, label, file_dir
        """
        img = Image.open(self.samples[idx][0])

        if self.transform:
            img = self.transform(img)

        return img, self.samples[idx][1], self.samples[idx][0]
