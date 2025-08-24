# Assignment 1 - Plant Classification

## Project Overview
This project focuses on building an automated system for classifying weeds and crop seedlings using image data. Early and accurate identification of weeds is essential for precision farming, as it reduces manual labor, lowers herbicide use, and supports sustainable agriculture.

The project compares three models:

- **Baseline CNN** – A standard Convolutional Neural Network with a Softmax classifier.
- **CNN + SVM** – Features extracted from CNNs are classified with a Support Vector Machine.
- **CNN + XGBoost** – CNN features are used to train an XGBoost classifier.

### Objectives
- Evaluate the effectiveness of CNNs and hybrid models (CNN+SVM, CNN+XGBoost) for agricultural image classification.
- Compare model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Identify the most effective approach in terms of accuracy, generalization, and computational efficiency.

### Outcomes
The study provides insights into whether replacing the CNN's Softmax classifier with SVM or XGBoost improves generalization. The results contribute to the development of precision farming technologies, aligning with UN Sustainable Development Goal (SDG) 2: Zero Hunger.

## Dataset Structure
The dataset is organized into training and testing sets with the following plant categories:

### Training Data (`train/`)
- **Black-grass** (260 images)
- **Charlock** (387 images)
- **Cleavers** (284 images)
- **Common Chickweed** (608 images)
- **Common wheat** (218 images)
- **Fat Hen** (472 images)
- **Loose Silky-bent** (651 images)
- **Maize** (218 images)
- **Scentless Mayweed** (513 images)
- **Shepherds Purse** (228 images)
- **Small-flowered Cranesbill** (493 images)
- **Sugar beet** (382 images)

### Testing Data (`test/`)
- Contains 379 unlabeled test images for model evaluation

## Files
- `COS801.ipynb` - Main Jupyter notebook containing the implementation
- `Merhawi_Hailu_u17095736_Assignment_1.pdf` - Assignment submission document
- `README.md` - This file

## Setup Instructions

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (install via pip):
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras opencv-python
  ```

### Getting Started
1. Clone or download this repository
2. Navigate to the project directory
3. Open `COS801.ipynb` in Jupyter Notebook
4. Run the cells sequentially to execute the analysis

## Project Structure
```
Assignment1/
├── COS801.ipynb                          # Main implementation notebook
├── README.md                             # Project documentation
├── Merhawi_Hailu_u17095736_Assignment_1.pdf  # Assignment submission
├── train/                                # Training dataset
│   ├── Black-grass/                      # Plant category 1
│   ├── Charlock/                         # Plant category 2
│   ├── Cleavers/                         # Plant category 3
│   ├── Common Chickweed/                 # Plant category 4
│   ├── Common wheat/                     # Plant category 5
│   ├── Fat Hen/                          # Plant category 6
│   ├── Loose Silky-bent/                 # Plant category 7
│   ├── Maize/                            # Plant category 8
│   ├── Scentless Mayweed/                # Plant category 9
│   ├── Shepherds Purse/                   # Plant category 10
│   ├── Small-flowered Cranesbill/        # Plant category 11
│   └── Sugar beet/                       # Plant category 12
└── test/                                 # Testing dataset
    └── [379 test images]
```

## Usage
1. **Data Loading**: The notebook loads training images from the `train/` directory
2. **Data Preprocessing**: Images are resized, normalized, and prepared for training
3. **Model Training**: A machine learning model is trained on the labeled training data
4. **Evaluation**: The model is evaluated on the test dataset
5. **Results**: Classification results and performance metrics are displayed

## Model Details
- **Input**: RGB images of plants
- **Output**: Classification into 12 plant categories
- **Architecture**: 
  - **Data Preprocessing**: Image resizing, normalization, and augmentation
  - **Feature Extraction**: Convolutional Neural Network (CNN) with multiple convolutional layers
  - **Classification Head**: Dense layers with dropout for regularization
  - **Activation Functions**: ReLU for hidden layers, Softmax for output layer
  - **Optimizer**: Adam optimizer with learning rate scheduling
  - **Loss Function**: Categorical Crossentropy for multi-class classification
  - **Regularization**: Dropout layers and early stopping to prevent overfitting
- **Performance Metrics**: 
  - **Accuracy**: Overall classification accuracy on test set
  - **Precision**: Per-class precision scores for each plant category
  - **Recall**: Per-class recall scores for each plant category
  - **F1-Score**: Harmonic mean of precision and recall for each class
  - **Confusion Matrix**: Visual representation of classification performance
  - **Top-1 Accuracy**: Percentage of correct top predictions
  - **Top-3 Accuracy**: Percentage of correct predictions in top 3 results
  - **Training Time**: Total time required for model training
  - **Inference Time**: Average time per image for prediction
  - **Model Size**: Total parameters and storage requirements

## Results
The plant classification models achieved the following performance across different architectures:

### Model Performance Comparison
| Model | F1_micro | ROC_AUC_micro |
|-------|-----------|----------------|
| **CNN & XGBoost** | **0.985263** | **0.999524** |
| CNN & SVM | 0.902105 | 0.997217 |
| Baseline CNN | 0.808421 | 0.983625 |

### Performance Analysis
- **Best Performing Model**: CNN & XGBoost ensemble achieves the highest F1-score (98.53%) and ROC-AUC (99.95%)
- **Hybrid Approach**: Combining CNN feature extraction with traditional ML classifiers shows significant improvement over baseline CNN
- **SVM Performance**: CNN & SVM achieves strong performance with 90.21% F1-score and 99.72% ROC-AUC
- **Baseline CNN**: Standalone CNN provides a solid foundation with 80.84% F1-score and 98.36% ROC-AUC

- #### **ROC Curve Analysis:**
- **CNN + XGBoost (Green Line)**: Perfect performance with AUC = 1.000
  - Achieves True Positive Rate of 1.0 with extremely low False Positive Rate
  - Best performing model for plant classification
  
- **CNN + SVM (Orange Line)**: Excellent performance with AUC = 0.997
  - Significantly outperforms baseline CNN
  - Strong classification capability with very low false positives
  
- **Baseline CNN (Blue Line)**: Very good performance with AUC = 0.984
  - Solid foundation model
  - Demonstrates the effectiveness of CNN feature extraction

#### **Performance Interpretation:**
- All models significantly outperform random classification (AUC = 0.5)
- Ensemble methods show clear improvement over standalone CNN
- CNN + XGBoost achieves near-perfect discrimination between plant classes
- The combination of deep learning features with traditional ML classifiers yields optimal results

### Key Findings
- Ensemble methods significantly outperform single models
- XGBoost classifier provides the best complement to CNN feature extraction
- All models achieve excellent ROC-AUC scores (>98%), indicating strong discriminative ability
- The hybrid CNN + XGBoost approach offers the optimal balance of accuracy and robustness

## Author
- **Student**: Merhawi Hailu
- **Student Number**: u17095736
- **Course**: COS801

## Notes
- Ensure all image files are properly loaded before running the notebook
- The dataset contains a total of 4,694 training images across 12 categories
- Test images are unlabeled and used for final model evaluation

## License
This project is created for educational purposes as part of the COS801 course assignment.

