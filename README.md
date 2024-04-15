 # Term Deposit Marketing

# Background:

Our objective is to create a robust machine learning system for a small startup, focusing mainly on providing machine learning solutions in the European banking market. Leveraging information coming from call center data, we are looking for ways to improve the success rate for calls made to customers for any product that the clients offer.

Ultimately, we are designing an ever-evolving machine learning product that offers high success outcomes while providing interpretability for our clients to make informed decisions.

# Data Description:
The dataset originates from the direct marketing initiatives of a European banking institution. The marketing campaign entails contacting customers through phone calls, often making multiple attempts to encourage product subscriptions, specifically for term deposits. Term deposits typically represent short-term financial commitments with maturities spanning from one month to a few years.

## Attributes:
 
age : age of customer (numeric)

job : type of job (categorical)

marital : marital status (categorical)

education (categorical)

default: has credit in default? (binary)

balance: average yearly balance, in euros (numeric)

housing: has a housing loan? (binary)

loan: has personal loan? (binary)

contact: contact communication type (categorical)

day: last contact day of the month (numeric)

month: last contact month of year (categorical)

duration: last contact duration, in seconds (numeric)

campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
     
# Goal(s):
Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
Success Metric(s): Hit 81% or above accuracy by evaluating with 5-fold cross validation, reporting the average performance score.
Bonus(es): Find the customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize. Find out What makes the customers buy? - Which feature should be the focuse be on.

# Solution:

The dataset has no duplicates or null values. It comprises 40,000 records with a variety of attributes. Notably, the dataset includes both binary and categorical attributes, providing a diverse set of information for analysis.
![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/a9d1fd79-86e9-4c84-acd0-46f76677ad68)

## Following the initial data preprocessing and EDA, the key findings are as follows:

* The class distribution in the target variable is imbalanced.
* We have 92.8% observations for non-term deposit and 30% observations for term deposit.

![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/49c1a5db-96bc-498b-9c84-d88a6cfa94b0)

## Target Variable y vs Age
![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/53da7947-7f1c-4286-9d61-d4ee4e09a24b)

* We can see that the median age of the customers subscribed to a term deposit is less than the median age of the customer not subscribed to a term deposit
* This shows that younger customers are more likely subscribe to a term deposit.
* There are outliers in boxplots of both class distributions
  
![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/bce0558d-a121-4325-9298-d97f31d16971)

## creating a series of box plots for different variables in your dataset using 

![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/b4ce979b-313a-46d1-aae6-a0896d3a6fd2)

# Data Preprocessing

## Outlier Detection

* The dataset contains outliers, particularly in the 'duration' and 'balance' attributes. We will retain outliers, given the imbalanced class nature. Additionally, normalization is skipped as it resulted in reduced model accuracy.


![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/d7c81a05-b530-4a51-80aa-3f828480cc1b)

![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/79243d3d-fa05-4914-b5ee-fd34e405d101)

* We performed one-hot encoding on all the categorical variables, converting them into a format suitable for machine learning algorithms.

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 40000 entries, 0 to 39999
Data columns (total 44 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   age                  40000 non-null  int64
 1   balance              40000 non-null  int64
 2   day                  40000 non-null  int64
 3   duration             40000 non-null  int64
 4   campaign             40000 non-null  int64
 5   job_admin            40000 non-null  bool 
 6   job_blue-collar      40000 non-null  bool 
 7   job_entrepreneur     40000 non-null  bool 
 8   job_housemaid        40000 non-null  bool 
 9   job_management       40000 non-null  bool 
 10  job_retired          40000 non-null  bool 
 11  job_self-employed    40000 non-null  bool 
 12  job_services         40000 non-null  bool 
 13  job_student          40000 non-null  bool 
 14  job_technician       40000 non-null  bool 
 15  job_unemployed       40000 non-null  bool 
 16  job_unknown          40000 non-null  bool 
 17  marital_divorced     40000 non-null  bool 
 18  marital_married      40000 non-null  bool 
 19  marital_single       40000 non-null  bool 
 20  education_primary    40000 non-null  bool 
 21  education_secondary  40000 non-null  bool 
 22  education_tertiary   40000 non-null  bool 
 23  education_unknown    40000 non-null  bool 
 24  housing_no           40000 non-null  bool 
 25  housing_yes          40000 non-null  bool 
 26  loan_no              40000 non-null  bool 
 27  loan_yes             40000 non-null  bool 
 28  contact_cellular     40000 non-null  bool 
 29  contact_telephone    40000 non-null  bool 
 30  contact_unknown      40000 non-null  bool 
 31  month_apr            40000 non-null  bool 
 32  month_aug            40000 non-null  bool 
 33  month_dec            40000 non-null  bool 
 34  month_feb            40000 non-null  bool 
 35  month_jan            40000 non-null  bool 
 36  month_jul            40000 non-null  bool 
 37  month_jun            40000 non-null  bool 
 38  month_mar            40000 non-null  bool 
 39  month_may            40000 non-null  bool 
 40  month_nov            40000 non-null  bool 
 41  month_oct            40000 non-null  bool 
 42  y_no                 40000 non-null  bool 
 43  y_yes                40000 non-null  bool 
dtypes: bool(39), int64(5)
memory usage: 3.0 MB

# Building and training Models

## Bagging Classifier

## Training performance:
Accuracy	Recall	Precision	F1
0	0.991625	0.999062	0.99055	0.994788

* Performance on training set varies between 0.925 to 0.928 recall.
* Let's check the performance on testing data.
  
## Testing performance:
Accuracy	Recall	Precision	F1
0	0.9915	0.99875	0.990701	0.994709

* Performance on testingg set varies between 0.998 to 0.928 recall.
  
![downloa![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/94978a9c-1229-4ca9-aaa8-17c5cf2484db)


![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/2b0f31cf-802c-4f71-a001-8ecfac31a3b2)

# Comparing all models till now
![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/2d1928cd-be1a-4ee9-a6d4-8e992708540a)

d](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/8046f5e0-03cc-4ff4-998d-1984eed0b022)

## Conclusion:
The analysis led to the following conclusions:

After evaluating seven different models, it is evident that their performance values significantly vary.

Accuracy: The bagging model achieved the highest accuracy value, followed by the Adaboost model, where as Xgboost, Decision Tree, and Random forest values were found to be overfitting.

The comparative analysis has been carried out by using 5-fold cross-validation methodology and the average score was found to be best for the Bagging Classifier model which was found to be 0.99.

Overfitting: It is important to note that while some models performed well on training data, they exhibited signs of overfitting when tested on the validation or test dataset. This means they may not generalize well to new data.

The Bagging Classifier model showed remarkable robustness, performing consistency well across various test scenarios and datasets, making it a strong contender in real-world deployment.

Business Context: Models that offer interpretability, such as the Bagging classifier model preferred in cases where understanding the model's decision-making process is crucial.

Ultimately, the choice of the best model is a Bagging Classifier and aligning with the given dataset characteristics, and constraints of the project. Further analysis, model fine tuning, or ensemble methods could be considered to improve the model’s performance.










