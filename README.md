# NYSE Sentiment Analsysis 

## Project Overview
The app uses the FinBERT NLP model to perform sentiment analysis on the inputted NYSE stock ticker. It retrieves news headlines relevant to the company using the NewsAPI, then calculates the average sentiment as an arbitrary value (between -5 and +5) for the last 14 days. It plots this on a graph comparing the change in sentiment over time to the change in stock price and caculates the Pearson Correlation Coefficient to measure how closely related the two variables are in order to determine how reliable news sentiment alone is at predicting stock change.

There is the option to download a csv of the news data and each articles appointed sentiment value, so the accuracy of the FinBERT model can be manually checked, along with the types of articles the api is returning.

**API Key:** The app prompts the user to input their [*NewsAPI*](https://newsapi.org) key.   
If you cant make an account use the following key: 4fb7451267354239a60d10f44e037b55  
[**Deployd App Link**](nysesentimentanalysis.up.railway.app)

## Tools
NLP Model: FinBERT (yiyanghkust/finbert-tone)  
Graphs: Plotly  
News & Stock Data: NewsAPI, yFinance  
Data Organisation & Calculations: Pandas, NumPy  
WebPage - Streamlit

## Results & Considerations
NewsAPI free tier has rate limits & limited coverage - so limited use and limited articles to analyse - resulting in sentiment being inaccurate on smaller companies due to limited news coverage.

**Pearson Correlation Coefficient**
Used to help mathematically understand how consistently the price & sentiment move together by calculating the strength and direction of their linear relationship.  

*A low coefficient does not necessarily mean a weak link between the two variables.*

- **Potential for Positive Bias:** Must consider that some articles report on recent stock price movements, which means sentiment can be a reaction to price changes & not just a predictor of them.  
- **Only Measures Trend Similarity:** The coefficient only takes into account how similar the graph trends are. Therefore, a consistently high but flat sentiment paired with a consistently growing stock price will have a low correlation, despite having a clear relationship.

Example ($PANW):
<img width="1304" height="520" alt="newplot-4" src="https://github.com/user-attachments/assets/b5b78239-7151-48eb-9772-873c68f8c328" />

**Flaws in FinBERT**  
Can misinterpret nuances in text leading to false positive or negative.
The model evaluated the article to be extremely negative, however the content suggests a more mixed view with both positive and negative signals.

| Headline | Description | Sentiment (-5 to +5) |
| --- | --- | --- |
Is investing in Apple stock now a no-brainer? | Discover why some experts say no stock is truly a no-brainer, as well as Apple's future investment outlook. | -4.99998825387905





