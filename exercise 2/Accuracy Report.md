## Analysis for Revenue Prediction

In this file, I summarize my work and thoughts along the analysis of this exercise. First glance at the exercise, 
I notice that it is a regression problem with some time effects (a panel to be specific). 

### Data Pre-processing
The first step is to inspect the data. Since the main objective is to predict for 2020 revenue data,
I consider the revenue data is more important than employee data. However, I spot some problems in
the raw dataset:

- Variable names have space and cause problems when retrieving;
- Revenue data is string since it contains comma;
- Some revenue data are negative;
- Companies have duplicate;
- A lot of Nan;
- Some companies have too few data points.

These problems are critical for prediction, I solve them as follows:

- Delete space
- Delete comma and convert string to float;
- Detect negative values and the corresponding companies' names, delete the data with such names;
- Keep the company with the most data in the duplicates, if they contain the same number of  data points,
keep the first one;
- Data imputation (will be mentioned later);
- To predict, my thought is that at least, a company must contain two data point for revenue (two points to
  draw a line). Therefore, I delete the companies if they only provide one data point.
  
After these actions, the dataset left me 13689 observations (originally it is 17424).

### Data Imputation

Too much Missing data! Simply delete it is not acceptable. Therefore, I apply some techniques to imputate for the
missing data. 

For employee missing data, one element should be considered. The number of employee could be affected by the that
in the previous. Some "big" companies' number of employees is relatively stable, but some "small" companies change a lot.
Therefore, we must consider at least 2 years' data around the missing data. The KNN imputer should be good choice. I use
the imputer methods provided by sklearn.

For revenue data, I simply use mean value of each company to fill the na, since I have filtered out the companies only
have one observation for revenue this way is doable. There are several reasons:

- Revenue is very important variable for prediction. I do not want to make it too sophisticated with some methods. I once
consider to use regression to fill the na. However, I will still use regression for prediction. So it is not a wise choice. 
  Besides, I did not take into account employee's number for imputate for revenue, since the employee number can change 
  very dramatically. I am not so sure what happen, but, given very little information about the dataset, a safe way is better.
  
- Since employee data is partly imputated. Looking at the missing data occupation pie chart, I decide not to use employee
data for imputate for revenue. 
  
The finally imputated data is in [Imputated Data.csv](https://github.com/Seaaann/Dealroom_intern/blob/main/exercise%202/Imputated%20Data.csv).
  
### Modeling

I choose OLS, Ridge and Random Forest to build a model. Before building the model, I did some easy feature engineering.
In the time-series prediction (it is not a very strict time-series problem though) problem, moving average is a very common
and useful predictor. I calculate the 3-year-moving-average and make two new variables. Besides, in time-series prediction,
we must take care of not to implicitly involve future information in order to avoid overfitting. Therefore, for example, if
we want to predict for revenue at t, we cannot involve any data at t and t+1.

I set my formula as:

Revenue_t = beta_1 * movingaverage_t-2 + beta_2 * movingaverage_t-1 + beta_3 * employee_t + e_t

for example:

Revenue_2019 = beta_1 * movingaverage_17-15 + beta_2 * movingaverage_18-16 + beta_3 * employee_2019 + e_2019

### Accuracy Analysis

Since we do not have the actual values for revenue_2020, I use revenue_2019 for tuning hyperparameters and cross-validation.
The full pipeline can be seen in [FeatureSelection.py](https://github.com/Seaaann/Dealroom_intern/blob/main/exercise%202/FeatureSelections.py).
The modeling result can be seen at [exercise 2.ipynb](https://github.com/Seaaann/Dealroom_intern/blob/main/exercise%202/exercise%202.ipynb).
The final results of the three models is in [regression_prediction.csv](https://github.com/Seaaann/Dealroom_intern/blob/main/exercise%202/regression_prediction.csv),
[ridge_prediction.csv](https://github.com/Seaaann/Dealroom_intern/blob/main/exercise%202/ridge_prediction.csv) and 
[rf_prediction.csv](https://github.com/Seaaann/Dealroom_intern/blob/main/exercise%202/rf_prediction.csv). The results are also
integrated into [Imputated Data.csv](https://github.com/Seaaann/Dealroom_intern/blob/main/exercise%202/Imputated%20Data.csv).

I find the best hyperparameters when tunning revenue_2019, and I use them for the forecast for revenue_2020. The predictors
involve MovingAverage_(2016-2018), MovingAverage_(2017-2019), Employee_(2020). I believe I use enough information and, at 
the same time, avoid noise. 

In the CV process, I present the results below:

| Evaluation Metrics/Models         | OLS               | Ridge       |  RandomForest    |
| ----------------------------------|:-----------------:| :----------:|-----------------:|
| Coefficient of determination      | 0.9328681         |  0.9329109  |    0.9243737     |
| Root Mean Square Log Error        | 0.882             |    0.807    |   0.956          |

The tuned hyperparameters:

| Parameters/Models                 | OLS               | Ridge       |  RandomForest    |
| ----------------------------------|:-----------------:| :----------:|-----------------:|
| fit_intercept                     | False             |  False      |         -        |
| alpha                             | -                 |  0.8367     |         -        |
| Solver                            | -                 |  'saga'     |         -        |
| bootstrap                         | -                 |  -          |         True     |
| max_features                      | -                 |  -          |    'sqrt'        |
| min_samples_leaf                  | -                 |  -          |         1        |
| min_samples_split                 | -                 |  -          |         2        |
| n_estimators                      | -                 |  -          |         400      |
| max_depth                         | -                 |  -          |         None     |


(Based on the limitation of computational power, I cannot try all the combination of parameters)


This result shows that Ridge regression should provide a better test result (since the actual revenue_2020 in unknown).

I think linear Ridge model in predicting revenue has some advantages:

- No clear non-linear patterns of company's revenue (given the time window is short).
- Unlike Lasso, Ridge takes all the factors into account (no dimension reducation needed).
- The penalized factor alpha should be able to reduce the residual sum of square, which generate a lower RMSLE as shown.






















