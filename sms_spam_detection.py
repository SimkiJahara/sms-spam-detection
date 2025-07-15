
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
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

X = data['message'].apply(preprocess)
y = data['label'].map({'ham': 0, 'spam': 1})

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)  # Sparse for most models
X_vec_dense = X_vec.toarray()  # Dense for GaussianNB, K-Means, t-SNE, Linear Regression, Perceptron

# Split and balance data
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

# t-SNE visualization
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train_bal, cmap='viridis')
plt.title('t-SNE Visualization of SMS Spam/Ham Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig(os.path.expanduser('~/sms_spam_project/tsne_plot.png'))
plt.close()

# Evaluate models
results = []

def evaluate_model(model, model_name, X_train=X_train_bal, X_test=X_test, y_train=y_train_bal, y_test=y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    if model_name == 'Linear Regression':
        y_pred = (model.predict(X_test) >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
    end_time = time.time()
    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'Recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        'Training Time (s)': end_time - start_time
    })

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_bal_dense)
cluster_labels = kmeans.predict(X_test_dense)
# Map clusters to spam/ham
cluster_mapping = {}
for cluster in [0, 1]:
    mask = (kmeans.labels_ == cluster)
    majority_label = y_train_bal[mask].mode()[0] if mask.sum() > 0 else 0
    cluster_mapping[cluster] = majority_label
kmeans_pred = [cluster_mapping[label] for label in cluster_labels]
results.append({
    'Model': 'K-Means',
    'Accuracy': accuracy_score(y_test, kmeans_pred),
    'Precision': precision_score(y_test, kmeans_pred, pos_label=1, zero_division=0),
    'Recall': recall_score(y_test, kmeans_pred, pos_label=1, zero_division=0),
    'F1-Score': f1_score(y_test, kmeans_pred, pos_label=1, zero_division=0),
    'Training Time (s)': time.time() - time.time()  # Approximate
})

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

for name, model in models:
    if name == 'KNN (PCA)':
        evaluate_model(model, name, X_train_pca, X_test_pca)
    elif name == 'KNN (t-SNE)':
        evaluate_model(model, name, X_train_tsne, X_test_tsne)
    elif name in ['Gaussian Naive Bayes', 'Linear Regression', 'Perceptron']:
        evaluate_model(model, name, X_train_bal_dense, X_test_dense)
    else:
        evaluate_model(model, name)

# Evaluate XGBoost
xgboost_model = XGBClassifier(eval_metric='logloss', random_state=42)
evaluate_model(xgboost_model, 'XGBoost')

# VotingClassifier (exclude LinearRegression and Perceptron)
voting_estimators = [(name, model) for name, model in models if name not in ['KNN (PCA)', 'KNN (t-SNE)', 'Linear Regression', 'Perceptron']]
voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
evaluate_model(voting_clf, 'Ensemble Voting', X_train_bal_dense, X_test_dense)

# Results
results_df = pd.DataFrame(results)
def classify_model(row):
    if row['F1-Score'] > 0.95 and row['Accuracy'] > 0.97:
        return 'High Performance'
    elif row['F1-Score'] > 0.90 and row['Accuracy'] > 0.95:
        return 'Moderate Performance'
    else:
        return 'Low Performance'

results_df['Performance Class'] = results_df.apply(classify_model, axis=1)
print("\nModel Performance Comparison:")
print(results_df.sort_values(by='F1-Score', ascending=False))

# F1-Score bar chart
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['F1-Score'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('F1-Score')
plt.title('F1-Score Comparison of SMS Spam Detection Models')
plt.tight_layout()
plt.savefig(os.path.expanduser('~/sms_spam_project/f1_score_plot.png'))
plt.close()

results_df.to_csv(os.path.expanduser('~/sms_spam_project/model_performance.csv'), index=False)

# Save best model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)
pickle.dump(model, open(os.path.expanduser('~/sms_spam_project/spam_model.pkl'), 'wb'))
pickle.dump(vectorizer, open(os.path.expanduser('~/sms_spam_project/vectorizer.pkl'), 'wb'))


