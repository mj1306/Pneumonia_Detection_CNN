# Bone Fracture Detection Using CNNs

This project utilizes **transfer learning** with pre-trained models (**VGG16** and **ResNet50**) to classify bone fractures from X-ray images. The objective is to determine which architecture yields better performance in detecting and classifying fractures.

## Project Overview

Bone fractures are common injuries, and timely, accurate diagnosis is crucial for effective treatment. This project applies deep learning techniques to classify X-ray images of bone fractures into 11 distinct categories. By leveraging pre-trained models like **VGG16** and **ResNet**, we aim to evaluate their performance in the task of fracture detection.

### Models Used:
- **VGG16** is a widely used Conolutional neural Network (CNN) with 16 layers (13 convolutional layers and 3 fully connected layers. Developed by the Visual Geometry Group, it's beauty lies in it's straight forward architecture and ease of application in transfer learning projects such as this one, making it very popular among researchers and for practical applications. If you're interested in learning more about the original architecture of the VGG16, Medium has an article on ["Everything you need to know about VGG16"](https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918).
- **ResNet50** is a deep CNN model that leverages residual (skip) connections to mitigate the vanishing gradient problem and allow for deeper Neural networks without hurting performance. The beauty in residual blocks is that unaltered inputs can be passed onto further layers; ensuring that essential information is preserved and propogated through the layers. Roboflow has a great article on the [Resnet50](https://blog.roboflow.com/what-is-resnet-50/#:~:text=ResNet%2D50%20is%20a%20convolutional,it%2C%20and%20categorize%20them%20accordingly).
## Dataset

The dataset consists of labeled X-ray images of bone fractures, each belonging to one of 11 fracture types. The dataset is created by kaggle and can be found here - [Bone Break Classification Dataset](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset)


