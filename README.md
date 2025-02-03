# Spam Mail Detection

## Overview
This project implements a **Spam Mail Detection** system using **Natural Language Processing (NLP)** techniques and **Machine Learning**. It classifies emails as either **spam** or **ham (not spam)** using the **Multinomial Naive Bayes (MNB)** algorithm. The model is trained using the **TF-IDF (Term Frequency-Inverse Document Frequency)** feature extraction technique.

## Features
- **Data Preprocessing**: Handling missing values and transforming categorical labels.
- **Feature Extraction**: Converting text data into numerical representations using **TfidfVectorizer**.
- **Model Training**: Using **Multinomial Naive Bayes (MNB)** to classify emails.
- **Performance Evaluation**: Analyzing model accuracy, classification report, and confusion matrix.
- **Hyperparameter Tuning**: Optimizing the model using **GridSearchCV**.
- **Prediction on Sample Emails**: Classifying user-input emails as spam or ham.

## Technologies Used
- **Python**
- **Pandas & NumPy** (Data Handling)
- **Scikit-learn** (ML Model, Metrics, and Feature Extraction)
- **Matplotlib & Seaborn** (Data Visualization)

## Dataset
The project uses a dataset containing labeled emails with two categories:
- **Ham (1):** Legitimate email
- **Spam (0):** Unwanted email

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-mail-detection.git
   cd spam-mail-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Upload the dataset (`mail_data.csv`) to your working directory.
4. Run the script:
   ```bash
   python spam_detection.py
   ```

## Code Explanation
### 1. Data Preprocessing
- Load dataset using `pandas`.
- Handle missing values.
- Convert categorical labels ('spam', 'ham') into binary format (1 for ham, 0 for spam).

### 2. Feature Extraction
- Use `TfidfVectorizer` to convert text data into numerical vectors.

### 3. Model Training
- Split the dataset into training (80%) and testing (20%) sets.
- Train a **Multinomial Naive Bayes** model on extracted features.

### 4. Performance Evaluation
- Calculate accuracy on both training and testing sets.
- Display a **classification report** and **confusion matrix**.
- Perform **hyperparameter tuning** using `GridSearchCV`.

### 5. Spam Mail Prediction
- Accepts custom email input and predicts whether it's spam or ham.
- Example:
   ```python
   input_your_mail = ["Congratulations! You have won a lottery. Call now to claim!"]
   input_data_features = feature_extraction.transform(input_your_mail)
   prediction = model.predict(input_data_features)
   print("Spam mail" if prediction[0] == 0 else "Ham mail")
   ```

## Results
- **Training Accuracy:** ~98%
- **Testing Accuracy:** ~96%
- **Confusion Matrix & Classification Report**: Included in output

## Visualization
- Confusion Matrix Heatmap
- Feature importance analysis using TF-IDF

## Future Improvements
- Implement **Deep Learning models** for better accuracy.
- Expand dataset for improved generalization.
- Deploy as a **web application** for real-time email classification.

## Contributing
Feel free to contribute by submitting **issues** or **pull requests**.

## License
This project is licensed under the **MIT License**.

---
**Author:** Your Name

For any queries, contact: [your-email@example.com](mailto:your-email@example.com)
