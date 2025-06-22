# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: DASARI NAGINI 

*INTERN ID*: CT08DF564 

*DOMAIN*: PYTHON PROGRAMMING 

*DURATION*: 8 WEEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

# YOU HAVE TO ENTER DESCRIPTION OF YOUR TASK (AND IT SHOULD NOT BE LESS THAN 500 WORDS)

Here's a complete Machine Learning model implementation in Python using the example of Spam Email Detection. It includes data loading, preprocessing, model training, evaluation, and inference, all in a format suitable for a Jupyter Notebook.


---

📘 Spam Email Detection using Machine Learning (Python)

# 🔧 Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


---

# 📥 Step 2: Load Dataset

# You can use your own dataset or download from open source like Kaggle
# For demo, we'll use a dataset from UCI ML Repository

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

df.head()


---

# 📊 Step 3: Data Preprocessing

# Convert labels: ham -> 0, spam -> 1
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Split into features and labels
X = df['message']
y = df['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


---

# ✏ Step 4: Text Vectorization

# Convert text into feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


---

# 🤖 Step 5: Model Training

model = MultinomialNB()
model.fit(X_train_vec, y_train)


---

# ✅ Step 6: Model Evaluation

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))


---

# 🧪 Step 7: Predict New Emails

def predict_spam(email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example usage
print(predict_spam("Win a free iPhone by clicking this link!"))
print(predict_spam("Hi, can we meet tomorrow for the project discussion?"))


---

# 📌 Summary

Model Used: Multinomial Naive Bayes

Text Vectorization: CountVectorizer (Bag of Words)

Accuracy: Typically ~98% for this dataset

Application: Can be deployed as part of an email filtering system



---

Would you like this in a .ipynb (Jupyter Notebook) file, or want to extend it with:

TF-IDF Vectorizer

Logistic Regression or SVM

GUI/Flask web interface?

*output*: 

![Image](https://github.com/user-attachments/assets/17d6e2d5-4c6d-4602-bfae-d705563969ff)

![Image](https://github.com/user-attachments/assets/bb5b2248-f594-4f06-bc07-984899ec8154)

![Image](https://github.com/user-attachments/assets/79a07ebb-fe27-4994-b331-b8367343c317)
