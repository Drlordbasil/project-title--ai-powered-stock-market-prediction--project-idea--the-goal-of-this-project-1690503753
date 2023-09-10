import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def collect_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start_date, end_date)
    return stock_data


def collect_news_sentiment(news_articles):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for article in news_articles:
        sentiment_scores.append(sid.polarity_scores(article))
    return sentiment_scores


def preprocess_data(stock_data, sentiment_scores):
    stock_data['Date'] = stock_data.index
    stock_data['Sentiment Score'] = [score['compound']
                                     for score in sentiment_scores]
    stock_data = stock_data.dropna()
    return stock_data


def create_input_features(stock_data):
    features = stock_data[['Open', 'High', 'Low', 'Volume', 'Sentiment Score']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features


def create_output_labels(stock_data, prediction_period):
    labels = stock_data['Close'].shift(-prediction_period)
    labels = labels[:-prediction_period]
    return labels


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model


def train_model(model, X_train, y_train):
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_split=0.2, callbacks=[early_stopping])


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def visualize_predictions(actual_prices, predicted_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices.index, actual_prices,
             label='Actual Prices', color='blue')
    plt.plot(predicted_prices.index, predicted_prices,
             label='Predicted Prices', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def generate_recommendation(prediction, threshold=0.03):
    if prediction >= threshold:
        return 'Buy'
    elif prediction <= -threshold:
        return 'Sell'
    else:
        return 'Hold'


# Set project parameters
stock_symbol = 'AAPL'
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2021, 12, 31)
prediction_period = 30

# Step 1: Data Collection
stock_data = collect_stock_data(stock_symbol, start_date, end_date)

# Step 2: Sentiment Analysis
news_articles = get_news_articles(stock_symbol, start_date, end_date)
sentiment_scores = collect_news_sentiment(news_articles)

# Step 3: Data Preprocessing
processed_data = preprocess_data(stock_data, sentiment_scores)

# Step 4: Feature Engineering
input_features = create_input_features(processed_data)
output_labels = create_output_labels(processed_data, prediction_period)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    input_features, output_labels, test_size=0.2, random_state=42)

# Step 6: Model Training
model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
train_model(model, X_train, y_train)

# Step 7: Model Evaluation
rmse = evaluate_model(model, X_test, y_test)
print(f'Root Mean Squared Error: {rmse}')

# Step 8: Visualization
predicted_prices = model.predict(X_test)
actual_prices = stock_data['Close'][-len(y_test):].reset_index(drop=True)
visualize_predictions(actual_prices, predicted_prices)

# Generate investment recommendation for the last date in the test set
last_prediction = predicted_prices[-1][0]
recommendation = generate_recommendation(last_prediction)
print(f'Investment Recommendation: {recommendation}')
