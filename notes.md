#7-27-2023
I think that first the program should train each ticker individually with a def train_ticker function rather than create a neural network that trains a large set of dependent variables (all of the tickers). This way I can more easily test each ticker with live data to create a "buy" or "sell" signal within the program.
Perhaps eventually I will want to train all long-type equities together where the individual biases also depends on this training set. That would of course be some later stage inclusion.

I also wanted to make note that there should be a feature where you input into the program that you have bought an asset, so that it may signal you when to sell, and not signal you to buy again until you have sold. I'm thinking that to save memory, and because this network will only be trained on data with frequency of at most 1d, I should split up the tickers into (8?) sets of equal size, each set being called once daily to signal a buy or a sell. An alternative way is to run each ticker over the course of 2-3pm to signal buy or sell (it will be an EOD signal).

With a combination of AlphaVantage and Nasdaq Data Link I can probably write a nice HFT deep learning algorithm. For now I'm just going to do daily frequency. With SerpAPI you could write a separate neural network that scrapes for news articles on a given a stock

#7-28-23
Use postman for debuggin and web scraping

#8-1-23
api.explodingtopics.com looks like it will be a very useful api (if it turns out to be free and functional)

I think that web scarping the popularity and sentiment of a given company at a given time should be the crux of the model

#8-3-23
For EPS data, I am using alphavantage and am including a counter of the number of weeks since the release of the data (because this might be relevant to price hype and what not)

#8-4-23
IMPORTANT: since I want to look at how the present day data effects the closing price of tomorrow, I have to copy the close column, and shift it down 1 position in a new row

#8-5-23
Ok, I got a trial model working properly, which is great. It also fits the data nicely, however, I was unsuccessful with producing a win rate that is consistently higher than that of just holding the stock. I need to try various loss functions and model parameters until I get something decent.

##HOW TO CREATE THE MODEL
Data Preprocessing: Clean and preprocess your data thoroughly. Handle missing values, scale the features appropriately, and consider normalization or standardization to make optimization easier for the neural network.

Feature Selection: If you have a large number of independent variables, consider performing feature selection techniques to identify the most relevant features for your model. Removing irrelevant or redundant features can help reduce computational complexity and improve model performance.

Windowing and Sequence Length: For time series data, create input sequences by using a sliding window approach. Determine an appropriate sequence length that captures temporal dependencies in your data without creating excessively long sequences that lead to memory issues.

Architecture Selection: Choose an appropriate neural network architecture for your time series problem. Common choices include recurrent neural networks (RNNs), long short-term memory (LSTM) networks, and gated recurrent units (GRUs). These architectures are designed to handle sequential data effectively.

Model Complexity: Be cautious with model complexity, especially when working with many independent variables. Deep neural networks with a large number of parameters can lead to overfitting, especially with limited data.

Regularization: Implement regularization techniques such as L1 or L2 regularization to prevent overfitting. Regularization can help control the model's complexity and improve its generalization ability.

Batch Normalization: Consider using batch normalization, especially in deep architectures. It can help stabilize training and accelerate convergence.

Learning Rate Scheduling: Implement learning rate scheduling to adaptively adjust the learning rate during training. This can help improve convergence and avoid overshooting the optimal solution.

Validation and Testing: Use appropriate validation and testing datasets to evaluate your model's performance and ensure it generalizes well to unseen data.

Hyperparameter Tuning: Experiment with different hyperparameters, such as the number of layers, units, learning rate, batch size, and dropout rate, to find the best configuration for your specific problem.

Memory Management: Be mindful of memory usage, especially if your dataset is large. Consider using mini-batch training to avoid loading the entire dataset into memory at once.

Model Interpretability: If model interpretability is crucial, consider using techniques such as attention mechanisms to understand which features contribute more to the predictions.

Ensemble Methods: For improved performance, consider using ensemble methods like bagging or stacking with multiple neural network models.