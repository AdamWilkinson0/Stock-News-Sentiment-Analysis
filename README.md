# NYSE Sentiment Analsysis 

## Project Overview
The app uses the FinBERT NLP model to perform sentiment analysis on the inputted NYSE stock ticker. It retrieves news headlines relevant to the company using the NewsAPI, then calculates the average sentiment as an arbitrary value (between -5 and +5) for the last 14 days. It plots this on a graph comparing the change in sentiment over time to the change in stock price and caculates the Pearson Correlation Coefficient to measure how closely related the two variables are in order to determine how reliable news sentiment alone is at predicting stock change.

There is the option to download a csv of the news data and each articles appointed sentiment value, so the accuracy of the FinBERT model can be manually checked, along with the types of articles the api is returning.

**API Key:** The app prompts the user to input their [*NewsAPI*](https://newsapi.org) key. If you cant make an account use the following key: 4fb7451267354239a60d10f44e037b55
[**Deployd App Link**](https://credit-risk-modeling-lauki-finance.streamlit.app/)

## Tools

## Results & Considerations




