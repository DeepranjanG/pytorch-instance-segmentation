# Custom instance segmentation pytorch

#### Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="python"/></a>
<a><img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/></a>
<a><img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
<a><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="opencv"/></a>
<a><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch"/></a>
<a><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)" alt="docker"/></a>
<a><img src="https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white" alt="gcp"/></a>
</p>

## Problem statement


## Solution Proposed
In this project, you have learned how to create your own training pipeline for instance segmentation models, on a custom dataset. 
For that, we wrote a torch.utils.data.Dataset class that returns the images and the ground truth boxes and segmentation masks. 
We also leveraged a Mask R-CNN model pre-trained on COCO train2017 in order to perform transfer learning on this new dataset.

## Dataset Used
 Penn-Fudan Database for Pedestrian Detection and Segmentation. 
 It contains 170 images with 345 instances of pedestrians, and we will use it to illustrate how to use the new features in torchvision in order to train an instance segmentation model on a custom dataset.

## Model Used
In this project, we will be using Mask R-CNN, which is based on top of Faster R-CNN. 
Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.

## How to run?

### Step 1: Clone the repository
```bash
git clone my repository 
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p env python=3.8 -y
```

```bash
conda activate env
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Install Google Cloud Sdk and configure

#### For Windows
```bash
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
```
#### For Ubuntu
```bash
sudo apt-get install apt-transport-https ca-certificates gnupg
```
```bash
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```
```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
```
```bash
sudo apt-get update && sudo apt-get install google-cloud-cli
```
```bash
gcloud init
```
Before running server application make sure your `Google Cloud Storage` bucket is available

### Step 5 - Run the application server
```bash
python app.py
```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build -t track . 
```

3. Run the Docker image

```
docker run -d -p 8080:8080 <IMAGEID>
```

4. Open docker image in interactive model

```
docker exec -ti <IMAGEID> bash
```

5. Authenticate GCloud

```
gcloud auth login
```

6. Authenticate default application

```
gcloud auth application-default login
```

👨‍💻 Tech Stack Used
1. Python
2. Pytorch
3. Docker

🌐 Infrastructure Required.
1. Google Cloud Storage
2. Google Compute Engine
3. Google Artifact Registry
4. Circle CI


## `src` is the main package folder which contains 

**Artifact** : Stores all artifacts created from running the application

**Components** : Contains all components of this project
- DataIngestion
- DataTransformation
- ModelTrainer
- ModelEvaluation
- ModelPusher

**Custom Logger and Exceptions** are used in the project for better debugging purposes.


## Conclusion



=====================================================================