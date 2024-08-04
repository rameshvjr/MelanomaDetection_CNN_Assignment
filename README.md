# Project Name
Build multiclass classification model using a custom convolutional neural network in TensorFlow. 

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)


## General Information

### Problem Statement:
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

### Main Objective of the project
The overarching goal is to support the efforts to reduce the death caused by skin cancer. The primary motivation that drives the project is to use the advanced image classification technology for the well-being of the people. Computer vision has made good progress in machine learning and deep learning that are scalable across domains.


### About the dataset used:
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

<b><a href=https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view?usp=sharing>Dataset is available here</a>

### The data set contains the following diseases:

    1. Actinic keratosis
    2. Basal cell carcinoma
    3. Dermatofibroma
    4. Melanoma
    5. Nevus
    6. Pigmented benign keratosis
    7. Seborrheic keratosis
    8. Squamous cell carcinoma
    9. Vascular lesion

### Project Pipeline Steps Involved:
1. Data Reading/Data Understanding → Defining the path for train and test images 
2. Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.
3. Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset 
4. Model Building & training : 
        Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
        Choose an appropriate optimiser and loss function for model training
        Train the model for ~20 epochs
        Write your findings after the model fit. You must check if there is any evidence of model overfit or underfit.
5. Chose an appropriate data augmentation strategy to resolve underfitting/overfitting 
6. Model Building & training on the augmented data :
        Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
        Choose an appropriate optimiser and loss function for model training
        Train the model for ~20 epochs
        Write your findings after the model fit, see if the earlier issue is resolved or not?
7. Class distribution: Examine the current class distribution in the training dataset 
        - Which class has the least number of samples?
        - Which classes dominate the data in terms of the proportionate number of samples?
8. Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
9. Model Building & training on the rectified class imbalance data :
        Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
        Choose an appropriate optimiser and loss function for model training
        Train the model for ~30 epochs
        Write your findings after the model fit, see if the issues are resolved or not?

### Key Components
#### Data Collection: 
Projects often use datasets from collaborations like the International Skin Imaging Collaboration (ISIC), which provides a large archive of skin images1.
#### Model Training: 
The collected images are used to train deep learning models. Data augmentation strategies are applied to prevent overfitting and improve model robustness1.
#### Evaluation: 
The models are evaluated on their ability to accurately classify images as melanoma or non-melanoma. Metrics like accuracy, sensitivity, and specificity are used to measure performance2.

### Benefits
#### Efficiency: 
Model can process images quickly, reducing the workload of dermatologists.
Accuracy: Model can potentially increase the accuracy and consistency of melanoma diagnoses.
Early Detection: These projects can help improve patient outcomes and survival rates.


## Conclusions

### Baseline Model

Accuracy and Loss charts for the baseline model

### Augmented Model

Accuracy and Loss charts for the augmented model

### Final Model

Accuracy and Loss charts for the final model

As the accuracy of the model increases, the loss decreases. The final model has an accuracy of 87% and a loss of 0.3. The model is able to predict the class of the lesion with a high accuracy.
Augmenting the data and countering class imbalance helped in improving the accuracy of the model.


## Technologies Used
- Python
- Tensorflow
- Keras
- Augmentor
- Matplotlib
- NumPy

## Acknowledgements
- This project was inspired by UpGrade curriculam
- References: Kaggle, Siva Kumar sir's session, Base line python script by Upgrad, online resources like kaggle, github, towardesdatascience.com, and others 
- This project is based on UpGrad's Melenoma Detection assignment as part of CNN.


## Contact
Created by [@rameshvjr] - feel free to contact me!

