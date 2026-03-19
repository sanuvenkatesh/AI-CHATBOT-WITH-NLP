"""
-----------------------------------------------
Name            : Sahana V
College         : [Seshadripuram College]
Internship Domain : Python Internship


Description:
This project is developed as part of my internship task.
It demonstrates the implementation of Python-based
solutions using libraries such as scikit-learn, NLTK,
and spaCy for building intelligent applications.

Technologies Used:
- Python
- Machine Learning
- Natural Language Processing
- VS Code
-----------------------------------------------
"""


"""
Ultimate NLP Chatbot using NLTK + spaCy + Wikipedia

Features
- NLP preprocessing
- Named Entity Recognition
- Sentiment detection
- Conversation history
- Persistent self-learning
- Context awareness (another joke)
- Joke counter
- Automatic Wikipedia answering
"""

import random
import json
import nltk
import spacy
import wikipedia
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Run: python -m spacy download en_core_web_sm")
    exit()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

chat_history = []
last_intent = None
joke_count = 0


# Knowledge Base


KNOWLEDGE_BASE = {

    "greet": {
        "patterns": ["hello", "hi", "hey", "greetings"],
        "responses": [
            "Hello! How can I help you?",
            "Hi there! What can I do for you today?"
        ]
    },

    "farewell": {
        "patterns": ["bye", "goodbye", "see you"],
        "responses": [
            "Goodbye! Have a nice day!",
            "See you later!"
        ]
    },

    "thanks": {
        "patterns": ["thanks", "thank"],
        "responses": [
            "You're welcome!",
            "Happy to help!"
        ]
    },

    "name": {
        "patterns": ["your name", "who are you", "what are you called"],
        "responses": [
            "I'm PyBot, an NLP chatbot built using Python."
        ]
    },

    "joke": {
        "patterns": ["joke", "funny", "laugh", "another"],
        "responses": [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the Python programmer go broke? Because he used up all his cache!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the computer go to the doctor? Because it had a virus!",
            "Why do Python programmers wear glasses? Because they can't C!"
        ]
    },

    "time": {
        "patterns": ["time", "clock"],
        "responses": ["__TIME__"]
    },

    "date": {
        "patterns": ["date", "today"],
        "responses": ["__DATE__"]
    },

    "python": {
        "patterns": ["python", "programming", "code"],
        "responses": [
            "Python is widely used in AI, machine learning, automation, and web development."
        ]
    }
}


# Memory functions


def load_learned_data():

    try:
        with open("learned_data.json","r") as f:
            KNOWLEDGE_BASE["learned"] = json.load(f)
    except:
        KNOWLEDGE_BASE["learned"] = {"patterns": [], "responses": []}


def save_learned_data():

    with open("learned_data.json","w") as f:
        json.dump(KNOWLEDGE_BASE["learned"], f, indent=4)



# NLP Processing


def preprocess(text):

    tokens = word_tokenize(text.lower())

    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in stop_words
    ]

    return tokens


def extract_entities(text):

    doc = nlp(text)

    return [(ent.text, ent.label_) for ent in doc.ents]



# Intent detection


def match_intent(user_input, tokens):

    user_text = user_input.lower()
    token_set = set(tokens)

    for intent, data in KNOWLEDGE_BASE.items():

        for pattern in data["patterns"]:

            pattern_tokens = pattern.split()

            if pattern in user_text:
                return intent

            if any(word in token_set for word in pattern_tokens):
                return intent

    return None



# Sentiment


def sentiment_response(user_input):

    sentiment = sia.polarity_scores(user_input)

    if sentiment["compound"] < -0.5:
        return "You seem upset. I'm here to listen."

    elif sentiment["compound"] > 0.6:
        return "You sound happy! That's great!"

    return None



# Wikipedia search


def search_wikipedia(query):

    try:

        summary = wikipedia.summary(query, sentences=2)

        return summary

    except wikipedia.DisambiguationError as e:

        return f"I found multiple results. Try being more specific like: {e.options[0]}"

    except:

        return None



# Response generator


def get_response(user_input):

    global last_intent
    global joke_count

    sentiment = sentiment_response(user_input)

    if sentiment:
        return sentiment

    tokens = preprocess(user_input)

    intent = match_intent(user_input, tokens)

    if intent is None and "another" in tokens and last_intent:
        intent = last_intent

    entities = extract_entities(user_input)

    if intent:

        last_intent = intent

        response = random.choice(KNOWLEDGE_BASE[intent]["responses"])

        if intent == "joke":
            joke_count += 1
            return f"Joke #{joke_count}: {response}"

        if response == "__TIME__":
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

        if response == "__DATE__":
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."

        if entities:

            entity_text = ", ".join([f"{e[0]} ({e[1]})" for e in entities])

            response += f"\n[Detected entities: {entity_text}]"

        return response

    # Try Wikipedia
    wiki_answer = search_wikipedia(user_input)

    if wiki_answer:
        return wiki_answer

    # Self learning
    print("PyBot: I don't know the answer. Can you teach me?")

    new_response = input("Teach me the correct response: ")

    KNOWLEDGE_BASE.setdefault("learned", {"patterns": [], "responses": []})

    KNOWLEDGE_BASE["learned"]["patterns"].append(user_input.lower())

    KNOWLEDGE_BASE["learned"]["responses"].append(new_response)

    save_learned_data()

    return "Thanks! I learned something new."



# Chat Loop


def main():

    load_learned_data()

    print("="*50)
    print("         PyBot - Intelligent NLP Chatbot")
    print("Type 'quit' to exit or 'history' for chat history")
    print("="*50)

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit","exit"]:
            print("PyBot: Goodbye!")
            break

        if user_input.lower() == "history":

            print("\nChat History")

            for speaker, msg in chat_history:
                print(f"{speaker}: {msg}")

            continue

        response = get_response(user_input)

        chat_history.append(("You", user_input))
        chat_history.append(("PyBot", response))

        print("PyBot:", response)


if __name__ == "__main__":
    main()