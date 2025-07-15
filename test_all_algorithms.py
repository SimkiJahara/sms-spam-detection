
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import os
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
try:
    data = pd.read_csv(os.path.expanduser("~/sms_spam_project/SMSSpamCollection.txt"), sep='\t', names=['label', 'message'], encoding='latin1')
except FileNotFoundError:
    print("Dataset not found. Please place SMSSpamCollection.txt in ~/sms_spam_project")
    exit(1)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# Load and preprocess data
X = data['message'].apply(preprocess)
y = data['label'].map({'ham': 0, 'spam': 1})

# Load vectorizer
try:
    vectorizer = pickle.load(open(os.path.expanduser('~/sms_spam_project/vectorizer.pkl'), 'rb'))
except FileNotFoundError:
    print("Vectorizer not found. Please run sms_spam_detection.py first.")
    exit(1)

# Vectorize and split data
X_vec = vectorizer.transform(X)
X_vec_dense = X_vec.toarray()
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
X_train_dense, X_test_dense = X_train.toarray(), X_test.toarray()
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
X_train_bal_dense = X_train_bal.toarray()

# Apply PCA and t-SNE for KNN
pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train_bal_dense)
X_test_pca = pca.transform(X_test_dense)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_train_tsne = tsne.fit_transform(X_train_bal_dense)
X_test_tsne = tsne.fit_transform(X_test_dense)

# Sample messages
messages = [
    "Free gift! Claim now at www.example.com",
    "Hey, let's meet for lunch tomorrow!",
    "Win $1000 now! Text WIN to 12345",
    "Reminder: Your meeting is at 2 PM"
]
messages_preprocessed = [preprocess(msg) for msg in messages]
X_vec_messages = vectorizer.transform(messages_preprocessed)
X_vec_messages_dense = X_vec_messages.toarray()
X_vec_messages_pca = pca.transform(X_vec_messages_dense)
tsne_sample = TSNE(n_components=2, random_state=42, perplexity=1)
X_vec_messages_tsne = tsne_sample.fit_transform(X_vec_messages_dense)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_bal_dense)
cluster_mapping = {}
for cluster in [0, 1]:
    mask = (kmeans.labels_ == cluster)
    majority_label = y_train_bal[mask].mode()[0] if mask.sum() > 0 else 0
    cluster_mapping[cluster] = majority_label

# Models
models = [
    ('Gaussian Naive Bayes', GaussianNB()),
    ('Complement Naive Bayes', ComplementNB()),
    ('SVM', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')),
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('KNN (PCA)', KNeighborsClassifier(n_neighbors=3, metric='cosine')),
    ('KNN (t-SNE)', KNeighborsClassifier(n_neighbors=3, metric='cosine')),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42)),
    ('Linear Regression', LinearRegression()),
    ('Perceptron', Perceptron(max_iter=1000, random_state=42))
]
xgboost_model = XGBClassifier(eval_metric='logloss', random_state=42)
voting_estimators = [(name, model) for name, model in models if name not in ['KNN (PCA)', 'KNN (t-SNE)', 'Linear Regression', 'Perceptron']]
voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')

# Test all models
for name, model in models + [('XGBoost', xgboost_model), ('Ensemble Voting', voting_clf)]:
    print(f"\n{name} Predictions:")
    if name == 'KNN (PCA)':
        model.fit(X_train_pca, y_train_bal)
        predictions = model.predict(X_vec_messages_pca)
    elif name == 'KNN (t-SNE)':
        model.fit(X_train_tsne, y_train_bal)
        predictions = model.predict(X_vec_messages_tsne)
    elif name in ['Gaussian Naive Bayes', 'Linear Regression', 'Perceptron']:
        model.fit(X_train_bal_dense, y_train_bal)
        predictions = (model.predict(X_vec_messages_dense) >= 0.5).astype(int) if name == 'Linear Regression' else model.predict(X_vec_messages_dense)
    elif name == 'Ensemble Voting':
        model.fit(X_train_bal_dense, y_train_bal)  # Use dense arrays
        predictions = model.predict(X_vec_messages_dense)
    else:
        model.fit(X_train_bal, y_train_bal)
        predictions = model.predict(X_vec_messages)
    for msg, pred in zip(messages, predictions):
        print(f"Message: {msg}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n")

# Test K-Means
print("\nK-Means Predictions:")
kmeans_pred = [cluster_mapping[label] for label in kmeans.predict(X_vec_messages_dense)]
for msg, pred in zip(messages, kmeans_pred):
    print(f"Message: {msg}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n")



