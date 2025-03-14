# **Spam Mail Detection with Streamlit Frontend**

## **Overview**
This project implements a **Spam Mail Detection** system using **Natural Language Processing (NLP)** and **Machine Learning**. It classifies emails as either **spam** or **ham (not spam)** using the **Multinomial Naive Bayes (MNB)** algorithm. The model is trained using **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.

Additionally, this project features an interactive **Streamlit**-based frontend for real-time email classification.

## **Features**
✅ **Data Preprocessing** – Handling missing values and transforming categorical labels  
✅ **Feature Extraction** – Converting text into numerical representations using **TfidfVectorizer**  
✅ **Model Training** – Using **Multinomial Naive Bayes (MNB)** for classification  
✅ **Performance Evaluation** – Analyzing accuracy, classification report, and confusion matrix  
✅ **Hyperparameter Tuning** – Optimizing the model with **GridSearchCV**  
✅ **Real-Time Prediction** – Classifying user-input emails as spam or ham using a **Streamlit frontend**  

## **Technologies Used**
- **Python**
- **Pandas & NumPy** – Data handling  
- **Scikit-learn** – ML model, feature extraction, and evaluation  
- **Matplotlib & Seaborn** – Data visualization  
- **Streamlit** – Frontend for real-time predictions  

## **Dataset**
The dataset contains labeled emails classified into two categories:  
📩 **Ham (1)** – Legitimate email  
🚫 **Spam (0)** – Unwanted email  

## **Installation & Setup**
1. **Clone the repository**  
   ```bash
   git clone https://github.com/AkshatNeolia/spam-mail-detection.git
   cd spam-mail-detection
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Upload the dataset** (`mail_data.csv`) to your working directory.  
4. **Run the backend script**  
   ```bash
   python spam_detection.py
   ```
5. **Run the Streamlit frontend**  
   ```bash
   streamlit run app.py
   ```

## **Code Explanation**
### **1. Data Preprocessing**
- Load the dataset using `pandas`.
- Handle missing values.
- Convert categorical labels ('spam', 'ham') into binary format (1 for ham, 0 for spam).

### **2. Feature Extraction**
- Use `TfidfVectorizer` to transform text into numerical features.

### **3. Model Training**
- Split the dataset into **training (80%)** and **testing (20%)** sets.
- Train a **Multinomial Naive Bayes** model on the extracted features.

### **4. Performance Evaluation**
- Compute **training & testing accuracy**.
- Generate a **classification report** and **confusion matrix**.
- Perform **hyperparameter tuning** using `GridSearchCV`.

### **5. Spam Mail Prediction** (Backend)
- Accepts custom email input and predicts whether it's spam or ham.  
Example:
   ```python
   input_your_mail = ["Congratulations! You have won a lottery. Call now to claim!"]
   input_data_features = feature_extraction.transform(input_your_mail)
   prediction = model.predict(input_data_features)
   print("Spam mail" if prediction[0] == 0 else "Ham mail")
   ```

### **6. Frontend (Streamlit UI)**
![Frontend Screenshot](![Screenshot 2025-03-14 221153](https://github.com/user-attachments/assets/845936a3-7d43-4d02-891e-85de0a153faa))  
- Users can enter an email in the **Streamlit interface** and get real-time predictions.
- Interactive UI with a simple, user-friendly design.
- Displays prediction results instantly.

## **Results**
📊 **Training Accuracy:** ~98%  
📈 **Testing Accuracy:** ~96%  
📌 **Confusion Matrix & Classification Report**: Generated in the output  

## **Visualization**
- 📌 **Confusion Matrix Heatmap**
- 🔍 **Feature Importance Analysis using TF-IDF**
