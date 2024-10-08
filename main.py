import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy
import PIL
from PIL import Image
import matplotlib.pyplot as plt 
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import preprocessing.VideoFrameDataset as VFD
import model.Encoder as E
import model.Loss as L

import argparse 



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="TCLR Anomaly Detection System")
parser.add_argument("training_data_path", type=str, help="Enter the complete path of your training data.")
parser.add_argument("num_frames", type=int, help="Enter the number of frames per video.")
parser.add_argument("-q", "--queue_size", type=int, default = 28, help="Enter the size for the temporal queue.")
parser.add_argument("-e", "--epochs", type=int, default = 20, help="Enter the number of epochs.")
parser.add_argument("-lr", "--learning_rate", type=float, default = 0.0001, help="Enter the learning rate.")
parser.add_argument("-f", "--queue_offset", type=int, default = 7, help="Enter the offset for the temporal queue.")
parser.add_argument("-t", "--temperature", type=float, default = 0.1, help="Enter the temperature value for the loss function.")
args = parser.parse_args()
root = args.training_data_path
num_frames = args.num_frames
BATCH_SIZE = 1
QUEUE_SIZE = args.queue_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
QUEUE_OFFSET = args.queue_offset
TAU = args.temperature



dataset = VFD.VideoFrameDatasetFlexible(root, 5, num_frames)
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False)



model = E.ResNet50Encoder()
model = model.to(device)
inst_disc_loss = L.PairwiseInstanceDiscrimination()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)



datapoints = np.empty((0, 64))
with torch.no_grad():
    for batch in dataloader:
        video = batch[0]
        for frame in video:
            frame = frame.unsqueeze(dim = 0)
            frame = frame.to(device)
            output = model(frame)
            datapoints = np.vstack([datapoints, output.detach().cpu().numpy()])



tsne_3D = TSNE(n_components=3, perplexity=30)
tsne_results_3D = tsne_3D.fit_transform(datapoints)



fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_results_3D[:, 0], tsne_results_3D[:, 1], tsne_results_3D[:, 2])
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.set_title("3D t-SNE Visualization of datapoints.")
plt.title("Before T-SNE : 3D")
fig.savefig('Before T-SNE : 3D.jpg')



tsne_2D = TSNE(n_components=2, perplexity=30)
tsne_results_2D = tsne_2D.fit_transform(datapoints)



fig = plt.figure(figsize=(8, 6))
plt.scatter(tsne_results_2D[:, 0], tsne_results_2D[:, 1])
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("2D t-SNE Visualization of datapoints.")
plt.title("Before T-SNE : 2D")
fig.savefig('Before T-SNE : 2D.jpg')



average_loss_list = []
for epoch in range(EPOCHS):

    print("Starting Epoch ", epoch, "...")
    
    batch_num = 0
    batchwise_loss_list = np.ndarray([])
    for batch in dataloader:

        print("Starting Batch ", batch_num, ":")

        queuewise_loss_list = np.ndarray([])
        video = batch[0]
        queue_index = 0
        while(queue_index + QUEUE_SIZE <= len(video)):
            temporal_queue = video[queue_index : queue_index + QUEUE_SIZE].to(device)
            non_temporal_queue = torch.cat([video[0 : queue_index], video[queue_index + QUEUE_SIZE : len(video)]], dim = 0).to(device)
            queue_embeddings = model(temporal_queue)
            outside_embedding = model(non_temporal_queue)
            loss = inst_disc_loss(queue_embeddings, outside_embedding, TAU)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            queuewise_loss_list = np.append(queuewise_loss_list, loss.detach().cpu().numpy())
            queue_index += QUEUE_OFFSET

            if(queue_index%14 == 0):
                print("Processing Queue: ", queue_index, "| Queue Loss = ", loss)
            
        average_loss_val_per_batch = np.mean(queuewise_loss_list)
        batchwise_loss_list = np.append(batchwise_loss_list, average_loss_val_per_batch)

        print("Finished processing batch: ", batch_num, "| Average Batchwise Loss = ", average_loss_val_per_batch)
        print()
        batch_num += 1

    average_loss_list.append(np.mean(batchwise_loss_list))

    print("Finished processing epoch: ", epoch, "| Average Epochwise Loss = ", np.mean(batchwise_loss_list))
    print()



plt.plot(range(len(average_loss_list)), average_loss_list, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.show()



datapoints = np.empty((0, 64))
with torch.no_grad():
    for batch in dataloader:
        video = batch[0]
        for frame in video:
            frame = frame.unsqueeze(dim = 0)
            frame = frame.to(device)
            output = model(frame)
            datapoints = np.vstack([datapoints, output.detach().cpu().numpy()])



tsne_3D = TSNE(n_components=3, perplexity=30)
tsne_results_3D = tsne_3D.fit_transform(datapoints)



fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_results_3D[:, 0], tsne_results_3D[:, 1], tsne_results_3D[:, 2])
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.set_title("3D t-SNE Visualization of datapoints.")
plt.title("Before T-SNE : 3D")
fig.savefig('Before T-SNE : 3D.jpg')



tsne_2D = TSNE(n_components=2, perplexity=30)
tsne_results_2D = tsne_2D.fit_transform(datapoints)



fig = plt.figure(figsize=(8, 6))
plt.scatter(tsne_results_2D[:, 0], tsne_results_2D[:, 1])
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("2D t-SNE Visualization of datapoints.")
plt.title("Before T-SNE : 2D")
fig.savefig('Before T-SNE : 2D.jpg')



model_state_dict = model.state_dict()
torch.save(model_state_dict, "saved_models/1_28_25_0.0001_7_0.1_ResNet-50.pth")
