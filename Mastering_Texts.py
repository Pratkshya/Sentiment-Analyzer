from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Sample data (positive & negative sentences)
train_data = [
    ("I am not mad", "definitely mad"),
    ("kei pani bhako chaina", "bhako cha for sure"),
    ("ma suteko aaba", "don't let her sleep"),
    ("I am having period cramps", "she needs pampering"),
    ("I don't want to talk about it", "press her to talk about it"),
    ("ma sanga aaba nabola", "please text her more"),
    ("I think you deserve better", "definitely is cheating"),
    ("I was busy", "no girl, he was entertaining someone else"),
     ("I don't care", "she cares a lot"),
    ("It's okay", "it's not okay"),
    ("Leave me alone", "don't leave her alone"),
    ("I am tired", "she needs attention"),
    ("hya", "please give her more attention"),
    ("chup", "ahahaha your call man"),
    ("k", "mad mad MAD"),
    ("eh", "very angry"),
    ("ehhhhhhh", "don't end the conversation"),
    ("khai", "she needs more attention"),
    ("aha j sukai gara", "the situation is getting tensed"),
    ("bola ki nabola", "please bolau"),
    ("eh game kheldai raichau bhare bolam na ta", "quit your game and give her the attention"),
    ("I am excited", "she wants you to share her excitement"),
    ("I am scared", "she needs you to reassure her"),
    ("I am confused", "she needs you to explain things to her"),
    ("I am frustrated", "she needs you to help her"),
    ("I am disappointed", "she needs you to make it up to her"),
    ("I am proud of you", "she wants you to feel appreciated"),
    ("I miss you", "she wants you to miss her too"),
    ("I love you", "she wants you to say it back")
]

# Prepare training data
texts, labels = zip(*train_data)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)

# Train a simple AI model (Naive Bayes Classifier)
model = MultinomialNB()
model.fit(X_train, labels)

# Function to predict sentiment
def predict_sentiment(text):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
    return render_template('main.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')