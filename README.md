# Project 4: Predicting Volatility Index price with News headlines

### Problem Statement

To train a classifer to predict whether Volatility Index(VIX) will go up or down and a sentimental analysis tool to predict the extent of the movement of Volatility Index based on our news headlines dataset which consists of top 25 news sentiments provided by Reddit WorldNews Channel for a period of 8 years. 

### Executive Summary

Efficient Market Hypothesis proposed by Fama [1965] states that stock market prices are driven by all observable information. In reality, it has been shown that investor sentiment can affect the asset prices due to the well-known psychological fact that investors with positive (negative) sentiment tend to make overly optimistic(pessimistic) judgments and decisions [Keynes, 1937]

#### 2 datasets : 

**Kaggle** : https://www.kaggle.com/aaron7sun/stocknews

Historical news headlines from Reddit WorldNews Channel (/r/worldnews). They are ranked by reddit users' votes, and only the top 25 headlines are considered for a single date. (Range: 2008-06-08 to 2016-07-01) All news are ranked from top to bottom based on how hot they are.Hence, there are 25 lines for each date. The first column is "Date", the second is "Label", and the following ones are news headlines ranging from "Top1" to "Top25".

**Violatiliy Index(VIX)**

Created by the Chicago Board Options Exchange (CBOE), the Volatility Index, or VIX, is a real-time market index that represents the market's expectation of 30-day forward-looking volatility. 

Derived from the price inputs of the S&P 500 index options, it provides a measure of market risk and investors' sentiments. It is also known by other names like "Fear Gauge" or "Fear Index." Investors, research analysts and portfolio managers look to VIX values as a way to measure market risk, fear and stress before they take investment decisions.

Higher market risk, fear and stress in the market usually indicates an increased in VIX price. 

**Classifier models**

Logistic Regression with CountVectorizer
Logistic Regression with TFID
Naive Bayes with TFID
Random Forest with TFID
3 layers of Stacked LSTM
LSTM with Convolutional Neural Network for Sequence Classification

The model that was found to be of the highest ROC AUC score is ____ (__%). Its sensitivity score is ____. This means that __% of wnvpresent cases were identified correctly. It will be helpful for the government to take further action.

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
---

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

2. Weather dataset
	* Distribution plots for 4 of the variables - tavg, dewpoint, preciptotal, wetbulb.
	tavg, dewpoint and wetbulb have left-skewed distributions while preciptotal has an overwhelming mode of 0.
	* Scatter plots of tavg, wetbulb and dewpoint against temperature.
	There is clear separation between tavg and dewpoint while there seems to be overlap between wetbulb with dewpoint and wetbulb with tavg, which may suggest correlation.

3. Spray dataset
	* Map of spray location and trap locations.
	Spray locations are centralised to specific clusters within the area covered by the trap locations.

---

### Modelling

1. Chose classifier models - K-nearest Neighbours, Logistic Regression, Random Forests, Decision Trees and ADA Boost and ran randomised search for hyperparameter tuning
2. Model Selection

The model was selected based on accuracy, ROC AUC and sensitivity. The accuracy of the production model is ____, compared to the baseline accuracy of 0.948.
The final model picked ....  We want to maximise sensitivity because the cost of predicting a false negative can lead to loss of lives.

---

### Conclusions and Recommendations

**Production Model**

As mentioned above, the production model would be **...**, which gives a rather high sensitivity rate of ..% .

**Limitation**

The limitation of this model lies in the use of time-series data which could mean that the data is autocorrelated. This can have implications on the computation of standard errors.

**Cost-Benefit Analysis**

*Costs*

We can quantify the costs of pesticide spraying by looking at the explicit costs associated with spray implementation.

The Chicago Department of Public Health (CDPH) implements mosquito spraying, using the chemical Zenivex™ E4 ,to reduce the occurrence of WNV. (City of Chicago, 2019). It is sprayed at 4.5 - 9 ounces per minute, at a vehicle speed of 10 - 15 mph (https://www.cmmcp.org/pesticide-information/pages/zenivex-e4-etofenprox).

Assuming each truck has an area of effect of around 3m to each side of the truck, the overall spray area is approximately 0.6 km2 per truck. The cost of Zenivex E4 is about \$80 USD per gallon. (http://www.gfmosquito.com/wp-content/uploads/2013/06/2013-North-Dakota-Bid-Tabulation.pdf)). Assuming a total spray duration of 5 hours, the cost of pesticides for each sprayer truck is \\$843.76 - \\$1687.50 USD </b>. Given that the total area of Chicago is 606.1 km2, it would take about 1000 trucks at the same time to cover the entire area. Hence, total costs  incurred could be $1,687,500.

Additional costs include the renting of trucks anad advertisment costs related to informing the public of the pesticide spraying.


*Benefits*

We can quantify the benefit of pesticide spraying by looking at the aversion of loss of income which could have resulted from WNV.

Benefits from mosquito spraying would include increased quality of life from fewer people falling sick and dying, increased workplace productivity from fewer people falling ill and going on medical leave, as well as savings in hospital expenses from treating WNV patients. Of these, only the latter two are measurable.

About 1 in 5 people infected with WNV develop West Nile fever with other symptoms such as headache, body aches, joint pains, vomiting, etc. Recovery from West Nile fever takes from a few days to several weeks, and prolonged fatigue is common (Peterson, 2019).

Though the majority of those infected will have mild symptoms or no symptoms at all, in some individuals, WNV can cause inflammation of the brain (encephalitis) and in severe cases, paralysis, coma or death.  The disease is most serious – even fatal – in those with compromised immune systems and the elderly. (https://westchicago.org/news/west-chicago-mosquito-abatement-district-2020-mosquito-season-update/)

The annual median household income in Chicago was \\$$57,238 (as of 2018; (https://datausa.io/profile/geo/chicago-il/) and can be used as an estimate to measure the loss of income during patient recovery from WNV.

In 2018, there were 176 WNV cases reported in Illinois where Chicago is located. https://www.cdc.gov/westnile/statsmaps/cummapsdata.html.

Given that 1 in 5 experience symptoms such as headache, body aches, joint pains, vomiting, diarrhea, or rash (https://www.cdc.gov/westnile/symptoms/index.html), this could mean that approximately 35 people out of 176 may have been hospitalized or received medical treatment. If these were working adults and required two weeks off work to recover, this would have resulted in a total loss of income of \\$2,200 per patient over 2 weeks or \\$77,000 for 35 patients. On average, each WNV patient spends approximately $33,143 per inpatient and ≈\\$6,317 per outpatient for all treatments. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3322011/). Therefore the total monetary loss caused by WNV in 2018 could be as high as \\$1,237,005.
