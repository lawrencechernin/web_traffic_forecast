# web_traffic_forecast

#Problem:
We have time series data of visit per page N days of historical data to forecast the next M days.
For the First Phase of the problem N=552. M=60 days.
For the second Phase we have data from July 1st 2015 through Sept 12 2017 ==> N=850

The submission to Kaggle is prediction of average page views for a given day of the M days, from Sept 13 => Nov 13

Best Models in the forums:

1) GroupBy which lumps together weekends and holidays for the most recent 49 days and calculates the median, the SMAPE=> 45.0
2) Fibonnaci windows using a given page and time series of N days to predict the N+1th day, the SMAPE => 44.8

A combination of the models gives 44.6 (web_traffic_forecast/blob/master/combined/two_models.py)
It is surprising that the combination gives better scores than each of the models, but I think behind the scenes Kaggle has been 
recalculating the scores. So previous scores might not be valid.

The limitations of these models:
1) replace the median function in the groupBy, but a function that considers the trend on the page or momentum
2) neither model is able to predict on a specific day during the M period.

Ways to improve:
Time series models that look at the shapes of the distributions. 
Neural Networks found hidden patterns
xgboost => but can we have 60 classes, one for each of the M days?

Training:
We can slice up the given training data of N days, into P+M<N, so we can build up large training sets this way.
For example: P=360, M=60, for July 1 2015, Dec 31 2015 used to predict Jan 1 to March 1.
There could be seasonal effects, so we may use month as a feature.


Features:
day of week
weekends
language of page
month of year
season
topic of page (business, sports, politics, etc...)
trends, eg ratio of past 10 days to past 30 days, other time bases features
device (mobile, web, etc...)
?

Limitations:
1) missing data in the training is not the same as having zero visits. Can we impute missing values?
