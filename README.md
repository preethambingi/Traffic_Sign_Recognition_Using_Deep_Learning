# Traffic_Sign_Recognition_Using_Deep_Learning

This project uses a deep learning model to classify images of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is built using Convolutional Neural Networks (CNN) and achieves high accuracy in recognizing various types of traffic signs, supporting applications in autonomous driving and driver assistance systems (ADAS).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

## Project Overview

The objective of this project is to classify traffic signs using a CNN model. The workflow consists of five main steps:
1. **Exploratory Data Analysis (EDA)** - Understand the dataset, analyze class distribution, and inspect image properties.
2. **Data Preprocessing** - Resize, normalize, and augment images to prepare them for training.
3. **Model Building** - Define and compile a CNN model suitable for image classification.
4. **Training** - Train the model on the preprocessed data.
5. **Evaluation** - Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset

The dataset used is the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news), which consists of:
- **43 classes** representing different types of traffic signs.
- **Over 50,000 images** captured in varying conditions, with each image labeled with a `ClassId`.

The dataset provides a realistic scenario for traffic sign recognition, making it ideal for training deep learning models for real-world applications.

### Downloading the Dataset

The GTSRB dataset can be downloaded from Kaggle using the following link: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

1. **Set Up Kaggle API**:
   - Go to your [Kaggle account settings](https://www.kaggle.com/account) and create a new API token under the **API** section. This will download a `kaggle.json` file.
   - Place `kaggle.json` in `~/.kaggle/` on your machine (or `/root/.kaggle/` if you're using Colab).
   - Set permissions for the file to ensure it is secure:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

2. **Download the Dataset Using Kaggle CLI**:
   - Run the following command to download the dataset:
     ```bash
     kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p /path/to/your/folder
     ```
   - Replace `/path/to/your/folder` with your desired folder path.

3. **Unzip the Dataset**:
   - After downloading, unzip the dataset:
     ```
     unzip /path/to/your/folder/gtsrb-german-traffic-sign.zip -d /path/to/your/folder
     ```

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
  - Analyze the distribution of classes to identify any class imbalance.
  - Inspect image sizes, aspect ratios, and pixel intensities to inform preprocessing decisions.

### 2. Data Preprocessing
  - **Resize** images to a fixed size (e.g., 30x30 pixels) to ensure consistency.
  - **Normalize** pixel values to a range of [0, 1].
  - **Augment** data with random transformations (e.g., rotation, shift) to enhance model robustness.

### 3. Model Building
  - Build a CNN model with multiple convolutional and pooling layers followed by fully connected layers.
  - Use dropout to reduce overfitting and improve generalization.

### 4. Training
  - Train the model using the preprocessed data and monitor validation accuracy to avoid overfitting.

### 5. Evaluation
  - Evaluate the model using test data and metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Requirements

Install the required packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```

### Key Dependencies
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `Pillow`
- `opencv-python`
- `scikit-learn`

## Setup and Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/preethambingi/Traffic_Sign_Recognition_Using_Deep_Learning.git
   cd Traffic_Sign_Recognition_Using_Deep_Learning
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   - Download the GTSRB dataset and ensure it is organized in the specified directory structure.

## Usage

Since the code is in a single Jupyter Notebook file (`Traffic_Sign_Recognition_Using_Deep_Learning.ipynb`), follow these steps to run it:

1. **Open the Notebook**:
   - Launch Jupyter Notebook and open `Traffic_Sign_Recognition_Using_Deep_Learning.ipynb`.
   
2. **Run All Cells**:
   - Execute the cells in the notebook sequentially to:
     - Perform EDA on the dataset.
     - Preprocess the data.
     - Build, train, and evaluate the CNN model.

## Results

The model achieved the following performance metrics on the test set:

- **Accuracy**: 0.9615
- **Precision**: 0.9637
- **Recall**: 0.9615
- **F1 Score**: 0.9615

### Classification Report

```
              precision    recall  f1-score   support

           0       0.98      0.97      0.97        60
           1       0.93      1.00      0.96       720
           2       0.96      0.97      0.96       750
           3       1.00      0.93      0.96       450
           4       1.00      0.95      0.97       660
           5       0.94      0.94      0.94       630
           6       0.98      0.97      0.98       150
           7       1.00      0.92      0.96       450
           8       0.92      0.99      0.95       450
           9       0.97      1.00      0.99       480
          10       1.00      1.00      1.00       660
          11       0.94      0.93      0.94       420
          12       1.00      0.96      0.98       690
          13       0.98      1.00      0.99       720
          14       1.00      1.00      1.00       270
          15       0.93      0.99      0.96       210
          16       0.99      1.00      0.99       150
          17       1.00      0.93      0.97       360
          18       1.00      0.81      0.89       390
          19       1.00      1.00      1.00        60
          20       0.81      0.99      0.89        90
          21       0.70      0.72      0.71        90
          22       0.99      0.83      0.90       120
          23       0.85      1.00      0.92       150
          24       0.99      0.99      0.99        90
          25       0.98      1.00      0.99       480
          26       0.91      0.99      0.95       180
          27       0.75      0.65      0.70        60
          28       0.96      0.98      0.97       150
          29       0.67      1.00      0.80        90
          30       0.77      0.68      0.72       150
          31       0.96      0.98      0.97       270
          32       0.98      1.00      0.99        60
          33       0.99      1.00      0.99       210
          34       0.99      1.00      1.00       120
          35       0.99      0.98      0.99       390
          36       1.00      1.00      1.00       120
          37       1.00      1.00      1.00        60
          38       0.99      1.00      1.00       690
          39       0.99      0.98      0.98        90
          40       0.95      0.98      0.96        90
          41       0.95      0.87      0.90        60
          42       0.88      0.98      0.93        90

   macro avg       0.94      0.95      0.94     12630
weighted avg       0.96      0.96      0.96     12630
```

## Future Work

- **Experiment with Deeper Models**: Try more complex architectures or pre-trained models (e.g., ResNet) to improve performance.
- **Enhanced Data Augmentation**: Experiment with more aggressive data augmentation to improve generalization.
- **Optimization for Real-Time Use**: Optimize the model for faster inference for real-time applications.
