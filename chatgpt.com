{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7dabe0e0",
   "metadata": {},
   "source": [
    "# Spam Email Detection - Machine Learning Model Implementation\n",
    "\n",
    "## 1. Import Required Libraries\n",
    "\n",
    "python\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eb8c8bef",
   "metadata": {},
   "source": [
    "## 2. Load Dataset\n",
    "\n",
    "python\n",
    "# Dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset\n",
    "df = pd.read_csv(\"spam.csv\", encoding='latin-1')[['v1', 'v2']]\n",
    "df.columns = ['label', 'message']\n",
    "df.head()\n",
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cd5c74c",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing\n",
    "\n",
    "python\n",
    "# Convert labels to binary\n",
    "df['label'] = df['label'].map({'ham': 0, 'spam': 1})\n",
    "\n",
    "# Check for null values\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Split dataset\n",
    "X = df['message']\n",
    "y = df['label']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2c160d6",
   "metadata": {},
   "source": [
    "## 4. Feature Extraction (Text Vectorization)\n",
    "\n",
    "python\n",
    "vectorizer = CountVectorizer()\n",
    "X_train_vec = vectorizer.fit_transform(X_train)\n",
    "X_test_vec = vectorizer.transform(X_test)\n",
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d371e2f",
   "metadata": {},
   "source": [
    "## 5. Train Naive Bayes Classifier\n",
    "\n",
    "python\n",
    "model = MultinomialNB()\n",
    "model.fit(X_train_vec, y_train)\n",
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1008a902",
   "metadata": {},
   "source": [
    "## 6. Model Evaluation\n",
    "\n",
    "python\n",
    "y_pred = model.predict(X_test_vec)\n",
    "\n",
    "# Accuracy\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.title(\"Confusion Matrix\")\n",
    "plt.show()\n",
    "\n",
    "# Classification Report\n",
    "print(classification_report(y_test, y_pred))\n",
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da7eba3e",
   "metadata": {},
   "source": [
    "## 7. Test with Custom Input\n",
    "\n",
    "python\n",
    "sample = [\"Congratulations! You've won a free ticket to Bahamas. Call now!\"]\n",
    "sample_vec = vectorizer.transform(sample)\n",
    "print(\"Prediction:\", \"Spam\" if model.predict(sample_vec)[0] else \"Ham\")\n",
    ""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2bd02e6",
   "metadata": {},
   "source": [
    "## ✅ Conclusion\n",
    "\n",
    "python\n",
    "print(\"The Naive Bayes classifier performs well for spam detection using text-based features.\")\n",
    ""
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
