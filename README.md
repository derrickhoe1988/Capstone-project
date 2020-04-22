# Project 4: Predicting Volatility Index price with News headlines

### Problem Statement

To train a classifer to predict whether Volatility Index(VIX) will go up or down and a sentimental analysis tool to predict the extent of the movement of Volatility Index based on our news headlines dataset which consists of top 25 news sentiments provided by Reddit WorldNews Channel for a period of 8 years. 

### Executive Summary

Efficient Market Hypothesis proposed by Fama [1965] states that market prices are driven by all observable information. In reality, it has been shown that investor sentiment can affect the asset prices due to the well-known psychological fact that investors with positive (negative) sentiment tend to make overly optimistic(pessimistic) judgments and decisions [Keynes, 1937]

Hence the purpose of this notebook is to check if the above hypthesis holds true and if the news sentiment can indeed affect the Volatilty price index and to what extent.

### Data Dictionary

**Top 25 news headlines from Reddit WorldNews Channel** : 

Data Source : https://www.kaggle.com/aaron7sun/stocknews

Historical news headlines from Reddit WorldNews Channel (/r/worldnews). They are ranked by reddit users' votes, and only the top 25 headlines are considered for a single date. (Range: 2008-06-08 to 2016-07-01) All news are ranked from top to bottom based on how hot they are.Hence, there are 25 lines for each date. The first column is "Date", the second is "Label", and the following ones are news headlines ranging from "Top1" to "Top25".

The columns included in this dataset are:

|S/N|Label|Description|
|---|:--|:--|
|1|ID|Numeric ID of the article|
|2|Title|the headline of the article|
|3|URL|URL of the article|
|4|PUBLISHER|Publisher of the article|
|5|CATEGORY|Category of the news item|
|6|STORY|Alphanumeric ID of the news story that the article discusses|
|7|HOSTNAME|Hostname where the article was posted|
|8|TIMESTAMP|Approximate timestamp of the article's publication, given in Unix time (seconds since midnight on Jan 1, 1970)|

### 2. Vix Historical data : VIX data for 2004 to 2020

**Violatiliy Index(VIX)** : 

Data Source : http://www.cboe.com/products/vix-index-volatility/vix-options-and-futures/vix-index/vix-historical-data

Created by the Chicago Board Options Exchange (CBOE), the Volatility Index, or VIX, is a real-time market index that represents the market's expectation of 30-day forward-looking volatility. 

Derived from the price inputs of the S&P 500 index options, it provides a measure of market risk and investors' sentiments. It is also known by other names like "Fear Gauge" or "Fear Index." Investors, research analysts and portfolio managers look to VIX values as a way to measure market risk, fear and stress before they take investment decisions.

Higher market risk, fear and stress in the market usually indicates an increased in VIX price. 

The columns included in this dataset are : 

|S/N|Label|Description|
|---|:--|:--|
|1|Date|Date of Vix|
|2|VIX Open|Opening price of VIX for the date|
|3|VIX High|Highest price of VIX for the date|
|4|VIX Low|Lowest price of VIX for the data|
|5|VIX Close|Closing price of VIX for the data|

**Classifier models**

***1st Model to predict if VIX is upordown : 

Logistic Regression with CountVectorizer
Logistic Regression with TFID
Naive Bayes with TFID
Random Forest with TFID
3 layers of Stacked LSTM

***2nd Model to predict the extent of VIX if it went up or down via sentiment analysis tool : 

Vader Sentiment Analysis
TextBlob Sentiment Analysis

---

### Directory Structure
```
project-4: Predicting Volatility Index price with News headlines
|__ code
|   |__ Data Manufacturing part 1.ipynb   
|   |__ EDA part 2.ipynb  
|   |__ Modelling part 3.ipynb
|   |__ Sentiment Analysis Tool part 4.ipynb
|   |__ Sentiment Analysis Tool part 5.ipynb
|__ data
|   |__ Combined_News_DJIA.csv
|   |__ final_dataframe.csv
|   |__ Final_df.csv
|   |__ news.csv
|   |__ out.csv
|   |__ vix_price.csv
|   |__ vixcurrent.csv
|   |__ X_features.csv
|__ Capstone.pdf
|__ README.md
```
---

### Data Dictionary

The columns included in this dataset are:


|S/N|Label|Description|
|---|:--|:--|
|1|ID|Numeric ID of the article|
|2|Title|the headline of the article|
|3|URL|URL of the article|
|4|PUBLISHER|Publisher of the article|
|5|CATEGORY|Category of the news item|
|6|STORY|Alphanumeric ID of the news story that the article discusses|
|7|HOSTNAME|Hostname where the article was posted|
|8|TIMESTAMP|Approximate timestamp of the article's publication, given in Unix time (seconds since midnight on Jan 1, 1970)|


### 2. Vix Historical data : VIX data for 2004 to 2020

Source : 
http://www.cboe.com/products/vix-index-volatility/vix-options-and-futures/vix-index/vix-historical-data

Content :

The columns included in this dataset are : 


|S/N|Label|Description|
|---|:--|:--|
|1|Date|Date of Vix|
|2|VIX Open|Opening price of VIX for the date|
|3|VIX High|Highest price of VIX for the date|
|4|VIX Low|Lowest price of VIX for the data|
|5|VIX Close|Closing price of VIX for the data|

### Data Cleaning

1. Kaggle Dataset 
	* dropped weekdends of the news headline for forex operating hours.
2. Violatiliy Index(VIX) Dataset
	* dropped weekdays to get date range from 2008-06-08 to 2016-07-01(Period of news dataset

### Pre-processing

1. Feature engineered the VIX Open and VIX close to create new column 'Upordown' where column is the difference between VIX Close and VIX OPEN

2. Merge Kaggle Dataset and Volatility Index(VIX) and kept the following columns:

Kaggle Dataset Columns : 

'Date', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7','Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15','Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23','Top24', 'Top25',

Volatility Index(VIX) Columns :
 
'upordown'

### Exploratory Visualisations

1. News Headlines dataset
	* Histogram plot of the news headlines to check for distribution of data from 2008 to 2016, this indicates an evenly balanced dataset across the timeframe. 
	* WordCloud created for year 2008 to 2016, this indicates the keys words for each year.

2. Volatility Index(VIX) dataset
	* plot a candlestick chart of Volatility Index price from year 2008 to 2016.
	* plot a chart to check how many days did VIX went up and how many days did VIX went down or stayed the same. We would found that during these period, 60% of the days , VIX either went down or stayed the same. Hence about 39% of the days, VIX closed higher than the opening price. 
    
---

### Modelling

#### Chose classifier models - 

**1. Price Direction Prediction (Up or Down) : LSTM3layers

Logistic Regression with CountVectorizer
Logistic Regression with TFID
Naive Bayes with TFID
Random Forest with TFID
3 layers of Stacked LSTM

**2. Sentiment Analysis (-1 Negative to +1 Positive) 

- Vader Sentiment Analysis
- TextBlob Sentiment Analysis

---

### Conclusions and Recommendations

#### Our Classifier

We have choose our final model for both : 

**1. Price Direction Prediction (Up or Down) : LSTM3layers

The summary of the results from the model above shows the performance of the various models and the different metrics used to evaluate the models, accuracy for the training and test dataset as well as ROC and AUC curve. The classifiers are sorted based on descending order of the sensitivity score on the validation dataset.

We choose the LSTM3layers as our best classifier, given that it has the highest AUC score as well as accuracy score for validation dataset (79%) and an accuracy of (76%) for our training dataset.

One possible reason why LSTMs performed better than the rest of the models was because they are very powerful in sequence prediction problems because they’re able to store past information. This is important in our case because some keywords in the previous news headlines is crucial in predicting the same news headlines to predict if Volatiliy index will fall or rise.

**2. Sentiment Analysis (-1 Negative to +1 Positive) : Text Blob Sentiment Analysis

Our score for Vader Sentiment Analysis achieved a accuracy of 39% overall and predicted 97% correctly when Vix prices goes up(value 1) and only able to predict 1.7% correctly when Vix goes down(value 0). On the other hand, the Textblob model predicted 31% correctly in the when VIX prices goes up and predicted 67% correctly when VIX goes down.

We have decided to chose TextBlob as our model due to the fact that it has higher accuracy score on dataset and that it has higher F1 score of 0.52 compared to Vader which achieve F1 score of only 0.24.

### Limitations

The limitation of this model lies in the assumption of efficient market hypothesis includes overconfidence,  overreaction, representative bias, and information bias that the new dataset is solely affecting the price direction and the volatility as well. There are several other factors that comes in play.

The dataset that i have collected may be news word headlines for a period of 8 years in reddit post however they also consist of non financial related news which in turns might not be a good predictor to the Volatility Index. 

### Area for further investigation

#### Change to a Fresh Dataset 

We have created a script to draw daily news from the NewsApi. News API is a simple HTTP REST API for searching and retrieving live articles from all over the web, in this case we have choosen to retrive the below  top news headlnes that are deemed more relevant to influecing the volatility index. 

News headlines consist of :

Top 10 BBC Headlines
Top 10 Google Headlines
Top 10 Tech Crunch Headlines
Top 20 Trump Headlines
Top 20 UK headlines
Top 20 US headlines
Source : https://newsapi.org/docs/endpoints/top-headlines

You can refer to the folder titled "script for website" , this script will be run on a daily basis to accumualte news headlines and to run the model to predict the price direction of volatility index. 

####  News reading 

We will proceed to generate top keywords based from the model so that users will read which news headlines are the causing more fear to the market, that can save them time to read through the news before they start trading the market. 

#### Deployment Phase

We have created a website for this predictors and will be looking to enhance it further.

New Website : https://derrickhoeyonghan4.wixsite.com/whatsnews

