# Hand-Gestures-Recognition-And-Segmentation

This GitHub repository contains code for a Convolutional Neural Network (CNN) model to detect five different hand gestures from segmented hand photos. The model is trained on a dataset using Keras with TensorFlow backend and includes data preprocessing, model building, training, and evaluation. Additionally, there are two separate files for real-time hand gesture recognition using OpenCV.

## Project Structure

The repository is organized as follows:

1. **Data Preparation:**
   - The dataset is stored in the "data" directory, with subdirectories for each class representing different hand gestures. You can download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/sarjit07/hand-gesture-recog-dataset/data) and extract it into the "data" directory.

2. **CNN Model:**
   - The initial model architecture is defined using Keras Sequential API in the main file.
   - The first attempt may show signs of overfitting, so a second model is created with dropout and regularization to address this issue.
   - The third model utilizes a pre-trained ResNet50 model for feature extraction, followed by additional layers for classification.

3. **Training and Evaluation:**
   - The models are trained using the training set and evaluated on the validation and test sets.
   - Training history, loss, and accuracy plots are visualized for each model.
   - Confusion matrices and classification reports provide detailed performance metrics.

4. **Real-time Hand Gesture Recognition:**
   - The "OpenCV_test" file demonstrates real-time hand gesture recognition using OpenCV with color-based segmentation based on blue gloves hands.
   - ![OpenCV_test](https://github.com/Sousannah/Hand-Gestures-Recognition-And-Segmentation/blob/main/Screenshot%202024-01-12%20184453.png)
   - The "seg" file performs real-time hand segmentation and classification.
   - ![seg](https://github.com/Sousannah/Hand-Gestures-Recognition-And-Segmentation/blob/main/color-based-screenshot.png)

5. **Dataset Testing:**
   - The "CNN02_Model_Test" file allows testing the trained model on any provided segmented hand photos.
   - Provide the directory path containing the segmented hand photos, and the model will predict the gestures.

## Instructions:

1. **Dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sarjit07/hand-gesture-recog-dataset/data) and extract it into the "data" directory.

2. **Training:**
   - Run the "CNN_Train" file to train and evaluate the CNN models.
   - Run the "segmentation_ResNet(one real data)" file to Train and evaluate the ResNet model on the data without doing data augmentation
   - Run the "segmentation_ResNet(one augmented data)" file to Train and evaluate the ResNet model on the data after doing data augmentation
   - Experiment with model architectures and hyperparameters to achieve optimal performance.

3. **Real-time Testing:**
   - Execute the "OpenCV_test" file for real-time hand gesture recognition using OpenCV with color-based segmentation based on blue gloves hands.
   - Execute the "seg" file for real-time hand segmentation and classification.

4. **Model Saving:**
   - The trained models are saved in the repository for later use.

## Requirements:

- Python 3.x
- Libraries: TensorFlow, Keras, OpenCV, scikit-learn, matplotlib, seaborn

Feel free to customize and extend the code according to your requirements. For any issues or suggestions, please create an issue in the [repository](https://github.com/Sousannah/hand-gestures-recognition-and-segmentation-using-CNN-and-OpenCV).

**Note: The trained ResNet models perform better than the CNN model**

Happy coding! 🚀
