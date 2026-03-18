import pandas as pd
import re
import string
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer

stop_words = set(['i','me','my','we','our','you','your','he','him','she','her',
'it','its','they','them','am','is','are','was','were','be','been','have','has',
'had','do','does','did','a','an','the','and','but','if','or','as','of','at',
'by','for','with','to','from','in','on','not','no','can','will','just','now',
'should','won','doesn','didn','isn','wasn','don'])

stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

real = [
"President signed climate bill into law.",
"Scientists discovered new deep sea fish.",
"Stock market rose following positive reports.",
"NASA launched satellite to monitor climate.",
"Government announced new spending plans.",
"Researchers found effective vaccine results.",
"Central bank raised rates to fight inflation.",
"Athletes broke world records at championship.",
"Study shows exercise improves mental health.",
"Tech firms reported strong quarterly earnings.",
"Water ice confirmed on moon surface.",
"Trade agreement signed by many countries.",
"Researchers found treatment for Alzheimers.",
"City council approved transportation project.",
"Doctors recommend exercise and healthy diet.",
"Parliament passed new education reform bill.",
"Astronomers found new planet in space.",
"Company announced record profits this year.",
"Engineers built bridge with green materials.",
"Leaders agreed on new climate targets.",
"Scientists made new drug to fight cancer.",
"Economy grew three percent last quarter.",
"Research shows benefits of healthy diet.",
"Hospital announced new heart disease cure.",
"Police caught suspects in bank robbery.",
"University launched artificial intelligence course.",
"Farmers reported record crop harvest season.",
"Airline announced new flights to Europe.",
"Scientists launched telescope for galaxies.",
"Government approved new energy projects.",
"New vaccine reduces risk of infection.",
"Doctors found new way to treat diabetes.",
"Scientists discovered cure for common cold.",
"Government built new schools in rural areas.",
"Researchers developed faster internet technology.",
"New law protects endangered animal species.",
"Scientists found new way to clean oceans.",
"Hospital opened new cancer treatment center.",
"Government launched free education program.",
"Scientists developed new water purification method.",
"New bridge connects two major cities.",
"Researchers found new renewable energy source.",
"Scientists developed new earthquake prediction system.",
"Government announced free healthcare for poor.",
"New study shows benefits of green tea.",
"Scientists discovered new antibiotic resistance cure.",
"Government launched new road safety campaign.",
"Researchers found new way to recycle waste.",
"Scientists developed faster electric car battery.",
"New treatment reduces recovery time after surgery.",
]

fake = [
"Aliens landed and government hiding truth.",
"Bleach cures diseases doctors hide this.",
"Moon landing faked in Hollywood studio.",
"Microchips implanted secretly through vaccines.",
"5G towers making people very sick.",
"Cancer cure hidden by big companies.",
"Earth is flat NASA always lying.",
"Bill Gates controlling world with money.",
"Chocolate makes you immune to virus.",
"Government adding chemicals to tap water.",
"Sharks swimming in city streets today.",
"Ancient city found on Mars secretly.",
"Celebrities replaced by robots say sources.",
"Time travel invented kept secret always.",
"Sun is alien spaceship in disguise.",
"Lemon water cures cancer overnight fast.",
"Government uses birds as spy drones.",
"Dinosaurs alive in underground secret base.",
"Onions on feet cure all diseases.",
"CIA invented internet to control minds.",
"Moon made of cheese say scientists.",
"World order secretly controls governments now.",
"Vaccines contain DNA that changes people.",
"Government hiding giant humans underground now.",
"Raw eggs every hour cures diabetes.",
"NASA hiding alien bases on moon.",
"Secret tunnels connect cities for elites.",
"Saltwater daily makes you live forever.",
"Pyramids built by aliens not humans.",
"Gravity is actually a government lie.",
"Drinking urine cures all known diseases.",
"Government spraying poison from airplanes daily.",
"Eating dirt boosts immunity say experts.",
"CIA controls all world leaders secretly.",
"Humans can live without food for years.",
"Government hiding cure for all cancers.",
"Aliens control all major world banks.",
"Drinking petrol cures joint pain doctors say.",
"Government using phones to read minds.",
"Scientists admit evolution is completely false.",
"Earth is only six thousand years old.",
"Government hiding technology that gives free energy.",
"Eating glass improves digestion say scientists.",
"CIA assassinated all major world leaders.",
"Aliens built all ancient wonders of world.",
"Government hiding giant underwater alien city.",
"Drinking acid removes all body toxins fast.",
"Scientists prove ghosts are real and visible.",
"Government using weather machines to cause disasters.",
"Eating rocks daily makes bones stronger.",
]

print("Real news count:", len(real))
print("Fake news count:", len(fake))

texts = real + fake
labels = [1]*len(real) + [0]*len(fake)

print("Total texts:", len(texts))
print("Total labels:", len(labels))

df = pd.DataFrame({'text': texts, 'label': labels})
df['clean'] = df['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1)
X_train_v = tfidf.fit_transform(X_train)
X_test_v = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_v, y_train)

y_pred = model.predict(X_test_v)
print("=" * 40)
print("Model Trained!")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("=" * 40)

def predict(text):
    v = tfidf.transform([preprocess(text)])
    p = model.predict(v)[0]
    c = model.predict_proba(v)[0][p] * 100
    return ("REAL" if p == 1 else "FAKE"), round(c, 1)

print("Type your news headline below.")
print("Type quit to exit.")
print("=" * 40)

while True:
    news = input("Enter headline: ")
    if news.lower() == 'quit':
        print("Goodbye!")
        break
    if news.strip():
        result, conf = predict(news)
        print("Result     :", result)
        print("Confidence :", conf, "%")
        print("-" * 40)
    else:
        print("Please enter a valid headline!")