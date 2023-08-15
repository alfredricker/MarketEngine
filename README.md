# Market Engine

## Summary
This project is a work in progress, but my main goal is to use PyTorch LSTM models as well as various web scraping techniques and data sourcing APIs in order to predict stock price movement on the order of days. Each stock is trained individually on its own price, sentiment, and popularity (Google trend) data in conjuction with general market data.

What is left to do is test the effectiveness of all the general market it data that I have collected, train a large variety of models and determine which performs the best, and implement live data on the model and recieve buy and sell signals.

## Market Data
### Federal Reserve Data

This data was sourced for free from a Quandl API on Nasdaq Data Link. For now, the data included in the neural network is quarterly CPI, Federal Funds rate, 3 month T-bill quotes, US quarterly GDP, the unemployment rate, M1 money supply, and M1 money velocity. I have yet to perform feature analysis to determine which signals influence daily price movement the most.

### Housing Data
I collected monthly housing market data from the free Zillow API. The data is from New York, Chicago, and LA and is expressed in monthly percent change.

### Equity Data
This is the data that is unique for each ticker's neural network. This data includes Google trend data, historical prices, volumes, earnings reports, earnings surprises, and relevant company financials.

## Sentiment Data
### News Sentiment
For this part of my model, I used web browsers' "inspect" feature to look for hidden APIs among notable market news outlets (Bloomberg, Reuters, Marketwatch). Using these APIs, BeautifulSoup, and Pandas, I was able to extract the headlines and summaries of thousands of articles dating back to 2015. I ran the headlines for each day through a pre-trained Roberta sentiment model, average the scores for that day, and included that data as a column in my training set.

### Retail Trading Sentiment
This data was collected through a Nasdaq Data link API. This API scans billions of dollars worth of daily trades and labels hundreds of stocks with a sentiment value, calculated from how much was bought and sold of a given stock on a given trading day.

## Technical Signals
### Momentum Signals
As of now, the model only includes the 14-day RSI momentum signal. I would like to perform feature analysis on a large number of technical signals and include those which have a significant influence on price movement into my model.
