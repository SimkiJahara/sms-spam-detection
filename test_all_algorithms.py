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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
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
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Apply PCA for KNN
pca = PCA(n_components=100, random_state=42)
X_train_knn = pca.fit_transform(X_train_bal.toarray())

# Sample messages
messages = [
    "Free gift! Claim now at www.example.com",
    "Hey, let's meet for lunch tomorrow!"
]
messages_preprocessed = [preprocess(msg) for msg in messages]
X_vec_messages = vectorizer.transform(messages_preprocessed)
X_vec_messages_knn = pca.transform(X_vec_messages.toarray())

# Models (same as sms_spam_detection.py)
models = [
    ('Naive Bayes', MultinomialNB()),
    ('SVM', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')),
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=3, metric='cosine')),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42))
]
xgboost_model = XGBClassifier(eval_metric='logloss', random_state=42)
voting_clf = VotingClassifier(estimators=models, voting='soft')

# Test all models
for name, model in models + [('XGBoost', xgboost_model), ('Ensemble Voting', voting_clf)]:
    print(f"\n{name} Predictions:")
    if name == 'KNN':
        model.fit(X_train_knn, y_train_bal)
        predictions = model.predict(X_vec_messages_knn)
    else:
        model.fit(X_train_bal, y_train_bal)
        predictions = model.predict(X_vec_messages)
    for msg, pred in zip(messages, predictions):
        print(f"Message: {msg}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n")