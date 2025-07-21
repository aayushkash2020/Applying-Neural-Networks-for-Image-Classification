# Applying Neural Networks for Image Classification
## Overview

This project explores a broad range of techniques for image classification and object recognition using neural networks. It progresses from basic image representations and MLP training to convolutional neural networks, transfer learning with pretrained models, and object detection using YOLO. The project also includes interpretability analyses and error diagnostics.

---

## Tasks Covered

### 1. Representation

- Explored fundamental representations of grayscale and RGB images.
- Visualized image data as pixel arrays and histograms.
- Converted pixel values to flattened vectors to enable basic classification with linear models.

### 2. MLP (Multilayer Perceptron)

- Implemented a simple fully connected neural network in PyTorch.
- Trained on the FashionMNIST dataset to classify images of clothing.
- Applied ReLU activations and trained with cross-entropy loss using the Adam optimizer.
- Visualized training/validation accuracy and loss using TensorBoard.

### 3. Recognition (Convolutional Neural Networks)

- Developed a CNN from scratch for classifying FashionMNIST images.
- Added convolutional layers, max pooling, and dropout for regularization.
- Compared performance against the MLP and showed improved generalization and test accuracy.
- Analyzed the effect of different kernel sizes and layer configurations.

### 4. Transfer (Transfer Learning with ResNet)

- Fine-tuned a pretrained ResNet-18 model on a custom mini-dataset of CIFAR-10 classes.
- Froze base layers and updated only the classifier head to adapt to the new task.
- Used data augmentation (random crops, flips, normalization) to reduce overfitting.
- Achieved high accuracy on new categories using very few training examples.

### 5. Detection (YOLO and Grad-CAM)

- Applied a pretrained YOLOv5 model for object detection on custom and sample images.
- Drew bounding boxes around detected objects with confidence scores.
- Evaluated detection performance across multiple categories.
- Used Grad-CAM to visualize which parts of the image contributed most to classification decisions, enhancing model interpretability.

---

## Results Highlights

- CNNs significantly outperformed MLPs on image recognition, demonstrating the power of spatial feature extraction.
- Transfer learning with ResNet-18 achieved strong results on new categories using minimal data.
- YOLOv5 detected multiple object categories in real-world scenes with high precision and recall.
- Grad-CAM revealed meaningful attention maps aligned with human intuition, validating model interpretability.

---

## Reflection

This project built my intuition for the strengths and weaknesses of different neural network architectures for vision tasks. I gained hands-on experience with end-to-end training in PyTorch, from defining models to tracking learning curves. Exploring transfer learning and object detection also helped me understand how state-of-the-art models can generalize to new domains. Finally, interpretability tools like Grad-CAM offered valuable insights into model behavior, reinforcing the importance of transparency in deep learning systems.

---

## Contact

For questions or collaboration, feel free to contact me at aayushkashyap2018@gmail.com.
