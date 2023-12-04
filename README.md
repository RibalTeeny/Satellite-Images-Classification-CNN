# Satellite Images Classification CNN

## About the Project

This project, developed during my internship at CGG from April to August 2023, focuses on the classification of satellite images using Convolutional Neural Networks (CNN). It is part of the "LandML" project, aimed at detecting illegal dump waste in satellite images, which is crucial for environmental protection. The project encompasses two main parts: developing an MLOps pipeline and improving performance of deep learning models suitable for image classification with a limited dataset.

### Organization

- **CGG**: A French geophysics company specializing in oil and gas exploration services.
  
### Role

- **Internship Position**: MLOps Engineer

## Project Overview

### Part One: MLOps Pipeline

- **Goal**: Develop a pipeline to automate, monitor, and manage machine learning models efficiently.
- **Implementation**: The pipeline involved downloading satellite images, preprocessing, data preparation, model training, and evaluation.
- **Technologies Used**: The project relied heavily on Kedro for data pipeline management and MLFlow for experimentation and model tracking.
- **Challenges**: Addressed issues related to dataset compatibility and network restrictions.
- **Result**: Created a flexible, reliable pipeline capable of running various components independently.

### Part Two: Improving Performance

- **Project LandML**: This project aimed at classifying RGB satellite images to detect illegal dump waste.
- **Data Handling**: Dealt with limited data availability, usage conflicts, and preprocessing challenges.
- **Models and Techniques**: Focused on ResNet50 FPN and VGG19, incorporating methods like weight decay and dropout to combat overfitting.
- **Loss Function and Optimizer**: Used Binary Cross Entropy with logits and the Adam optimization algorithm.
- **Pretraining and Fine-tuning**: Explored the impact of pre-training [ImageNet] on high-resolution data and subsequent fine-tuning on low-resolution data.

### Key Contributions

- Developed a robust MLOps pipeline for efficient handling of machine learning operations.
- Conducted comprehensive tests on deep learning models, enhancing their performance on limited datasets.

## Conclusion

This internship experience at CGG, particularly within the Incubator branch, provided me with invaluable insights into the application of deep learning techniques in real-world scenarios.
