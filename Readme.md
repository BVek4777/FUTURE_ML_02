📊 Stock Price Prediction Using LSTM
This project is part of the ML Internship by FutureIntern and focuses on building a machine learning model using LSTM (Long Short-Term Memory networks) to predict future stock prices.

🎯 Objective
To develop a machine learning model based on historical stock market data to estimate future stock prices.
The project aims to apply deep learning techniques to capture temporal patterns in time series data for accurate stock forecasting.

🛠 Tools & Technologies
Python — Core programming language
Pandas — Data manipulation and analysis
NumPy — Numerical computing
Scikit-learn — Preprocessing, model evaluation, and scaling
TensorFlow / Keras — Deep learning model (LSTM)
yfinance — Fetching real-time historical stock data
Matplotlib — Data visualization

📚 Project Workflow
1. Data Collection
Use the yfinance API to download historical stock price data (e.g., closing prices).

2. Data Preprocessing
Handle missing values (if any).
Normalize the data using MinMaxScaler.
Create input sequences (e.g., use 60 previous days to predict the next day).
Split the dataset into training and testing sets.

3. Model Building
Build an LSTM model using the Keras Sequential API.

Architecture:
One or more LSTM layers
Dropout layers for regularization
Dense output layer for prediction

4. Model Compiling
Compile the model with a suitable optimizer, loss function, and evaluation metrics.

5. Model Training
Train the model and monitor both training and validation loss.

6. Model Evaluation
Evaluate using metrics like Mean Squared Error (MSE) and R² Score.
Visualize loss curves (training vs. validation).

7. Prediction & Visualization
Forecast stock prices on the test set.
Plot actual vs. predicted prices for visual evaluation.

📈 Predict and Visualize Unseen Data (Next n Days)
Predict stock prices for the next n days.
Visualize the predicted prices using line plots.



