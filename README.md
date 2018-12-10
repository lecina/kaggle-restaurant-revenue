# kaggle-restaurant-revenue
https://www.kaggle.com/c/restaurant-revenue-prediction

First attempt into Kaggle.

The training dataset consists of only 137 entries with 37 columns of obfuscated 
data with not apparent missing numbers. It is crucial to avoid overfitting with
such a poor amount of data. We are not given any explanation of whether they are 
numerical or categorical. Besides, we have the opening date, city name, city group,
and restaurant type.

The basic approach has been correcting positive skews taking log1p; transforming
the opening date into an "age" (e.g. days or months since open), gathering unpopulated
cities/restaurant types together.

We used a bunch of linear and non-linear algorithms and trained them using a 
gridsearch. In the linear regressions, we removed coefficients such that 
|coefficient| < 0.25, in order to avoid overfitting.

Finally, we combined three different linear regressions, obtained with CV with
n=3,4, and 5, and three gradient boosted regressions, obtained in the same way.
The combination was naive (average), but fancier strategies could have been tried.
