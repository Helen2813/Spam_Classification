import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC

# Чтение данных
df = pd.read_csv('spam.tsv', sep='\t')

# Балансировка данных
ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']

ham = ham.sample(spam.shape[0])
data = pd.concat([ham, spam], ignore_index=True)

# Удаляем пропуски
data.dropna(subset=['message', 'label'], inplace=True)

# Проверка совпадения длины
print(data['message'].shape, data['label'].shape)

# Разделение на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.3, random_state=0, shuffle=True
)

# Random Forest
classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
print('Random Forrest Accuracy: ', rf_accuracy)
print('Random Forrest Confusion Matrix: ', confusion_matrix(y_test, y_pred))
print('Random Forest: ', classification_report(y_test, y_pred))

# SVM
svm = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', SVC(C=100, gamma='auto'))
])
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_score(y_test, y_pred)
print('SVM Accuracy: ', accuracy_score(y_test, y_pred))
print('SVM Confusion Matrix: ', confusion_matrix(y_test, y_pred))
print('SVM: ', classification_report(y_test, y_pred))