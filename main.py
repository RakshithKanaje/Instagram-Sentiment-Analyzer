import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from sklearn.model_selection import train_test_split
import datetime
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
from textblob import TextBlob
from playwright.sync_api import sync_playwright
import getpass

def setup_database():
    try:
        print("Connecting to MongoDB...")
        client = MongoClient("mongodb://localhost:27017/")
        db = client["instagram_analytics"]
        collection = db["posts"]
        print("Connected to MongoDB successfully!")
        return collection
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        sys.exit()

def collect_instagram_posts(page, hashtag, max_posts=50):
    print(f"Collecting posts for hashtag: {hashtag}")
    scraped_posts = []

    try:
        search_url = f"https://www.instagram.com/explore/tags/{hashtag}/"
        page.goto(search_url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_selector("article div img", timeout=60000)

        print("Starting to scrape posts...")

        for _ in range(10):
            page.evaluate("window.scrollBy(0, window.innerHeight);")
            time.sleep(3)
            posts = page.query_selector_all("article div img")
            for img in posts:
                img_src = img.get_attribute("src")
                alt_text = img.get_attribute("alt") or ""

                if "profile picture" in alt_text.lower():
                    continue
                
                if img_src and img_src not in [p['image_url'] for p in scraped_posts]:
                    scraped_posts.append({
                        "image_url": img_src,
                        "caption": alt_text,
                        "timestamp": datetime.datetime.now(),
                        "hashtag": hashtag
                    })
                    print(f"Scraped: {img_src}")

                    if len(scraped_posts) >= max_posts:
                        break
            if len(scraped_posts) >= max_posts:
                break

    except Exception as e:
        print(f"Error during collection: {e}")
    return scraped_posts

def process_posts(posts):
    print("Processing posts - Analyzing sentiment...")
    for post in posts:
        blob = TextBlob(post["caption"])
        post["sentiment_score"] = blob.sentiment.polarity
        post["sentiment"] = "positive" if blob.sentiment.polarity > 0 else "negative" if blob.sentiment.polarity < 0 else "neutral"
        print(f"Processed Sentiment for: {post['image_url']} -> Score: {post['sentiment_score']} Sentiment: {post['sentiment']}")
    return posts

def store_posts_in_db(posts, collection):
    print("Storing posts into MongoDB...")
    try:
        if posts:
            collection.insert_many(posts)
            print(f"Successfully stored {len(posts)} posts into MongoDB!")
        else:
            print("No posts to store.")
    except Exception as e:
        print(f"Error storing posts to MongoDB: {e}")

def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=64))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_sentiment_model(collection):
    print("Fetching data from MongoDB...")
    posts = list(collection.find())
    captions = [post["caption"] for post in posts]
    sentiments = [
        2 if post["sentiment"] == "positive" else 1 if post["sentiment"] == "neutral" else 0
        for post in posts
    ]

    print("Applying tokenization and padding...")
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)

    sentiments_encoded = to_categorical(sentiments, num_classes=3)

    x_train, x_val, y_train, y_val = train_test_split(padded_sequences, sentiments_encoded, test_size=0.2)

    print("Initializing and training the model...")
    model = build_model()
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    print("Saving the model...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/instagram_sentiment_model_tf.h5"
    if os.path.exists(model_path):
        version = 1
        while os.path.exists(f"models/instagram_sentiment_model_v{version}.h5"):
            version += 1
        model_path = f"models/instagram_sentiment_model_v{version}.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}.")

    print("Saving training metrics plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    metrics_path = 'training_metrics.png'
    plt.savefig(metrics_path)
    print(f"Training metrics saved as {metrics_path}.")

def instagram_pipeline(username, password, hashtag, max_posts=50):
    collection = setup_database()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720}
        )
        page = context.new_page()

        try:
            print("Logging in to Instagram...")
            page.goto("https://www.instagram.com/accounts/login/", timeout=60000)
            page.wait_for_selector('input[name="username"]', timeout=30000)
            page.fill('input[name="username"]', username)
            page.fill('input[name="password"]', password)
            page.click("button[type='submit']")
            page.wait_for_url("https://www.instagram.com/", timeout=60000)
            print("Login successful!")

            page.set_default_navigation_timeout(120000)

            raw_posts = collect_instagram_posts(page, hashtag, max_posts)

            processed_posts = process_posts(raw_posts)

            store_posts_in_db(processed_posts, collection)

            train_sentiment_model(collection)

        except Exception as e:
            print(f"Pipeline encountered an error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    username = input("Enter your Instagram username: ")
    password = getpass.getpass("Enter your Instagram password: ")
    hashtag = input("Enter the hashtag you want to analyze: ")
    max_posts = int(input("Enter the maximum number of posts to scrape: "))
    instagram_pipeline(username, password, hashtag, max_posts)
