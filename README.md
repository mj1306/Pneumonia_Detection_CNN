# Pneumonia Detection Using CNNs

This project utilizes **transfer learning** with pre-trained models (**VGG16** and **ResNet50**) to classify whether a patient has pneumonia from X-ray images. The objective is to determine which architecture yields better performance in detecting and classifying fractures.

## Project Overview

Pneumonia is a serious respiratory condition that requires timely and accurate diagnosis for effective treatment. This project applies deep learning techniques to classify chest X-ray images as either **Normal** or **Pneumonia**. By leveraging pre-trained models like **VGG16** and **ResNet50**, we aim to evaluate their performance in detecting pneumonia with high accuracy and reliability.  


### Models Used:

- **VGG16** is a widely used Convolutional neural Network (CNN) with 16 layers (13 convolutional layers and 3 fully connected layers). Developed by the Visual Geometry Group, its beauty lies in it's straight forward architecture and ease of application in transfer learning projects such as this one, making it very popular among researchers and for practical applications. If you're interested in learning more about the original architecture of the VGG16, Medium has an article on ["Everything you need to know about VGG16"](https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918).
- **ResNet50** is a deep CNN model that leverages residual (skip) connections to mitigate the vanishing gradient problem and allow for deeper neural networks without hurting performance. The beauty in residual blocks is that unaltered inputs can be passed onto further layers; ensuring that essential information is preserved and propagated through the layers. Roboflow has a great article on the [Resnet50](https://blog.roboflow.com/what-is-resnet-50/#:~:text=ResNet%2D50%20is%20a%20convolutional,it%2C%20and%20categorize%20them%20accordingly).
## Dataset

The dataset consists of labeled X-ray images of chest x-rays classified as normal/pneumonia. The dataset is created by kaggle and can be found here - [Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download)

## Objectives and Evaluation  

- **Objective:** To build a robust deep learning model capable of accurately classifying chest X-ray images into two categories: Normal and Pneumonia.  
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1 Score will be used to evaluate the model's performance on the validation and test datasets.

## Why This Project is Relevant  

This project demonstrates the applications of Convolutional Neural Networks in the medical imaging domain, including:  
- The use of **Transfer Learning** to learn useful patterns recognized by pre-trained models.  
- Importance of **Fine-tuning** CNN architectures to adapt your model to generalize well on your target dataset.  

## Acknowledgments  

This project was inspired by the need for improved diagnostic tools in medical imaging. Special thanks to the authors of the [Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download) dataset.  


