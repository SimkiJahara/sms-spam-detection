import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import time
import pickle
import os

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
X_vec = vectorizer.fit_transform(X)

# Split and balance data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Apply PCA for KNN
pca = PCA(n_components=100, random_state=42)
X_train_knn = pca.fit_transform(X_train_bal.toarray())
X_test_knn = pca.transform(X_test.toarray())

# Evaluate models
results = []

def evaluate_model(model, model_name, X_train=X_train_bal, X_test=X_test):
    start_time = time.time()
    model.fit(X_train, y_train_bal)
    y_pred = model.predict(X_test)
    end_time = time.time()
    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label=1),
        'Recall': recall_score(y_test, y_pred, pos_label=1),
        'F1-Score': f1_score(y_test, y_pred, pos_label=1),
        'Training Time (s)': end_time - start_time
    })

# Models (excluding XGBoost from VotingClassifier)
models = [
    ('Naive Bayes', MultinomialNB()),
    ('SVM', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')),
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=3, metric='cosine')),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

for name, model in models:
    if name == 'KNN':
        evaluate_model(model, name, X_train_knn, X_test_knn)
    else:
        evaluate_model(model, name)

# Evaluate XGBoost separately
xgboost_model = XGBClassifier(eval_metric='logloss', random_state=42)
evaluate_model(xgboost_model, 'XGBoost')

# VotingClassifier without XGBoost
voting_clf = VotingClassifier(estimators=models, voting='soft')
evaluate_model(voting_clf, 'Ensemble Voting')

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
results_df.to_csv(os.path.expanduser('~/sms_spam_project/model_performance.csv'), index=False)

# Save best model (Random Forest, based on previous results)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)
pickle.dump(model, open(os.path.expanduser('~/sms_spam_project/spam_model.pkl'), 'wb'))
pickle.dump(vectorizer, open(os.path.expanduser('~/sms_spam_project/vectorizer.pkl'), 'wb'))