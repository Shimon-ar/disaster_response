import sys
import pandas as pd
import nltk
from sqlalchemy import create_engine
import string
import joblib
import time

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# nltk.download(['punkt', 'wordnet', 'stopwords', 'maxent_ne_chunker', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * from disaster', engine)

    X = df['message']
    Y = df[df.columns[5:]]

    return X, Y, Y.columns


def lemmatize(token, tag):
    lemmatizer = WordNetLemmatizer()

    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)

    return lemmatizer.lemmatize(token, tag)


def tokenize(text):
    clean_tokens = []

    for sent in sent_tokenize(text):
        for token, tag in nltk.pos_tag(word_tokenize(text)):

            token = token.lower().strip().strip('_').strip('*')

            # If stopword, ignore token and continue
            if token in stopwords.words('english'):
                continue

            # If punctuation, ignore token and continue
            if all(char in string.punctuation for char in token):
                continue

            # Lemmatize the token and yield
            clean_tokens.append(lemmatize(token, tag))

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('m_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__max_features': (None, 10000, 30000)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_t = y_pred.T

    for index, col in enumerate(Y_test):
        print(category_names[index])
        print(classification_report(Y_test[col], y_pred_t[index]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model {}...'.format(time.ctime()))
        model.fit(X_train, Y_train)

        print('finished train model at : {}'.format(time.ctime()))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
