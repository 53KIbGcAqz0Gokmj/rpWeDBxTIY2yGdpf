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









