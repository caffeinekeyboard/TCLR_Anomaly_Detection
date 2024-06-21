
# Temporal Contrastive Learning for Anomaly Detection in Video Data.

This project explores a new approach towards temporal contrastive learning in video data, using a novel framework which uses 65% less parallel computing resources. The design of the framework is completely novel, and utilises new concepts such as temporal queue and accumulating losses.

This project experiments with ResNet-18 and ResNet-50 as encoder architectures. A private proprietary dataset was used for this project, consisting of over 100 hours of CCTV surveillance. The project was inpired from Ishan Dave's framework for temporal contrastive learning.


## Performance:

The dataset consisted of 8 videos with 700 frames each, amounting to a total of 5600 individual datapoints. The hyperparameter values are same as the default arguement values listed in the latter sections.\
The following are the T-SNE plots representing the data in a latent space of 64 dimensions before and after our best training session:

Before Training:
![Before Training](https://github.com/caffeinekeyboard/TCLR_Anomaly_Detection/assets/96489029/ce015ff9-c2ab-460c-8c59-aee01197ebd3)

![Before Training](https://github.com/caffeinekeyboard/TCLR_Anomaly_Detection/assets/96489029/ebb4e9f1-463d-4ca9-ade7-a6a1dd7cb2fc)


After Training:
![After Training](https://github.com/caffeinekeyboard/TCLR_Anomaly_Detection/assets/96489029/13d2bff4-025f-4952-950f-42d2e44d9b69)

![After Training](https://github.com/caffeinekeyboard/TCLR_Anomaly_Detection/assets/96489029/7184b5cd-3d16-4bd9-8edd-1618aff9dff3)

These are the computing resources utilized after maximum optimization:
|Encoder Architecture|Maximum VRAM Utilised|GPU Utilised|
|---|---|---|
|ResNet-50| 15.77 GB | NVIDIA-Tesla P100 (Kaggle)|
|ResNet-18| 7.81 GB | NVIDIA RTX 3070-Ti Laptop GPU|

## Run Locally:

Run the following commands to experiment with temporal queues with your own data:

```bash
    git clone https://github.com/caffeinekeyboard/Dog_Emotion_Classification.git
```
```bash
    python main.py /path/to/your/training/data/ <number_of_frames_per_video> 
```  

The following are the defaults for optional arguements:

|Arguement|Purpose|Default|
|---|---|---|
|-q|Size of the temporal queue|28|
|-e|Number of epochs|20|
|-lr|Learning Rate|0.0001|
|-f|Offset of the temporal queue.|7|
|-t|Temperature for Pairwise-InstDisc Loss (Tau)|0.1|

## Related:

[TCLR by Ishan Dave](https://github.com/DAVEISHAN/TCLR)\
[Dave et. al. Research Paper](https://arxiv.org/abs/2101.07974)\
[Instance Discrimination Loss](https://arxiv.org/abs/2103.15916)

