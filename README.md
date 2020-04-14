# Project 4: Predicting Volatility Index price with News headlines

### Problem Statement

To train a classifer to predict whether Volatility Index(VIX) will go up or down and a sentimental analysis tool to predict the extent of the movement of Volatility Index based on our news headlines dataset which consists of top 25 news sentiments provided by Reddit WorldNews Channel for a period of 8 years. 

### Executive Summary

Efficient Market Hypothesis proposed by Fama [1965] states that stock market prices are driven by all observable information. In reality, it has been shown that investor sentiment can affect the asset prices due to the well-known psychological fact that investors with positive (negative) sentiment tend to make overly optimistic(pessimistic) judgments and decisions [Keynes, 1937]

Hence the purpose of this notebook is to check if the above hypthesis holds true and if the news sentiment can indeed affect the Volatilty price index and to what extent.

#### 2 datasets : 

**Kaggle** : https://www.kaggle.com/aaron7sun/stocknews

Historical news headlines from Reddit WorldNews Channel (/r/worldnews). They are ranked by reddit users' votes, and only the top 25 headlines are considered for a single date. (Range: 2008-06-08 to 2016-07-01) All news are ranked from top to bottom based on how hot they are.Hence, there are 25 lines for each date. The first column is "Date", the second is "Label", and the following ones are news headlines ranging from "Top1" to "Top25".

**Violatiliy Index(VIX)**

Created by the Chicago Board Options Exchange (CBOE), the Volatility Index, or VIX, is a real-time market index that represents the market's expectation of 30-day forward-looking volatility. 

Derived from the price inputs of the S&P 500 index options, it provides a measure of market risk and investors' sentiments. It is also known by other names like "Fear Gauge" or "Fear Index." Investors, research analysts and portfolio managers look to VIX values as a way to measure market risk, fear and stress before they take investment decisions.

Higher market risk, fear and stress in the market usually indicates an increased in VIX price. 

**Classifier models**

1st Model to predict if VIX is upordown : 

Logistic Regression with CountVectorizer
Logistic Regression with TFID
Naive Bayes with TFID
Random Forest with TFID
3 layers of Stacked LSTM
LSTM with Convolutional Neural Network for Sequence Classification

2nd Model to predict the extent of VIX if it went up or down via sentiment analysis tool : 

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

#### 1. Chose classifier models - 

##### For Price Direction Prediction (Up or Down) 

Logistic Regression with CountVectorizer, Logistic Regression with TFID, Naive Bayes with TFID, Random Forest with TFID,3 layers of Stacked LSTM,LSTM with Convolutional Neural Network for Sequence Classification

The model was selected based on accuracy, ROC AUC and sensitivity. The accuracy of the production model is ____, compared to the baseline accuracy of 0.948.
The final model picked ....  We want to maximise sensitivity because the cost of predicting a false negative can lead to loss of lives.

##### For Price Volatility Prediction (Range -1(Negative) to 1(Positive))

Vader Sentiment Analysis
TextBlob Sentiment Analysis

The model was selected based on accuracy, ROC AUC and sensitivity. The accuracy of the production model is ____, compared to the baseline accuracy of 0.948.
The final model picked ....  We want to maximise sensitivity because the cost of predicting a false negative can lead to loss of lives.

---

### Conclusions and Recommendations

**Production Model**

As mentioned above, the production model would be **...**, which gives a rather high sensitivity rate of ..% .

**Limitation**

The limitation of this model lies in the use of time-series data which could mean that the data is autocorrelated. This can have implications on the computation of standard errors.
