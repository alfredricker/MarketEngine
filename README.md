# Market Engine

## Summary
This project is a work in progress, but my main goal is to use PyTorch LSTM models as well as various web scraping techniques and data sourcing APIs in order to predict stock price movement on the order of days. Each stock is trained individually on its own price, sentiment, and popularity (Google trend) data in conjuction with general market data.

What is left to do is test the effectiveness of all the general market it data that I have collected, train a large variety of models and determine which performs the best, and implement live data on the model and recieve buy and sell signals.

## Data Used
### Federal Reserve Data

This data was sourced for free from a Quandl API on Nasdaq Data Link. For now, the data included in the neural network is quarterly CPI, Federal Funds rate, 3 month T-bill quotes, US quarterly GDP, the unemployment rate, M1 money supply, and M1 money velocity.

### Housing Data
I collected monthly housing market data from the free Zillow API. The data is from New York, Chicago, and LA and is expressed in monthly percent change (when fed to the model).

### Equity Data
This is the data that is unique for each ticker's neural network. This data includes Google trend data, historical prices, volumes, earnings reports, earnings surprises, and relevant company financials.
