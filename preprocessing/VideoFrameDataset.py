import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models



class VideoFrameDataset(Dataset):
    """
        Input(root): Directory path with surveillance videos.\n
        Output: PyTorch Dataset with each video represented as an iterable, with first 700 frames loaded into a torch tensor.
    """

    def __init__(self, root, divisions):
        self.divisions = divisions
        self.root = root
        self.transforms = transforms.ToTensor()
        self.path_list = os.listdir(root)
        for i in range(0, len(self.path_list)):
            self.path_list[i] = self.root + '/' + self.path_list[i]
    

    def __len__(self):
        return len(self.path_list*self.divisions)
    

    def __getitem__(self, index):
        video_number = index//self.divisions
        frame_set = index%self.divisions
        video_path = self.path_list[video_number]
        frame_list = []
        video = cv2.VideoCapture(video_path)
        for i in range(0, 700):
            _, frame = video.read()
            frame = Image.fromarray(frame)
            frame_list.append(self.transforms(frame))
        frame_list = np.array(frame_list)
        frame_list = torch.from_numpy(frame_list)
        return frame_list[(frame_set*(700//self.divisions)) : (frame_set*(700//self.divisions) + (700//self.divisions))]
    

class VideoFrameDatasetFlexible(Dataset):
    """
        Input(root): Directory path with surveillance videos.\n
        Output: PyTorch Dataset with each video represented as an iterable, with first 700 frames loaded into a torch tensor.
    """

    def __init__(self, root, divisions, num_frames):
        self.num_frames = num_frames
        self.divisions = divisions
        self.root = root
        self.transforms = transforms.ToTensor()
        self.path_list = os.listdir(root)
        for i in range(0, len(self.path_list)):
            self.path_list[i] = self.root + '/' + self.path_list[i]
    

    def __len__(self):
        return len(self.path_list*self.divisions)
    

    def __getitem__(self, index):
        video_number = index//self.divisions
        frame_set = index%self.divisions
        video_path = self.path_list[video_number]
        frame_list = []
        video = cv2.VideoCapture(video_path)
        for i in range(0, self.num_frames):
            _, frame = video.read()
            frame = Image.fromarray(frame)
            frame_list.append(self.transforms(frame))
        frame_list = np.array(frame_list)
        frame_list = torch.from_numpy(frame_list)
        return frame_list[(frame_set*(self.num_frames//self.divisions)) : (frame_set*(self.num_frames//self.divisions) + (self.num_frames//self.divisions))]