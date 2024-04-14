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
    	Output (Label): y - has the client subscribed to a term deposit? (binary)
     
# Goal(s):
Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
Success Metric(s): Hit 81% or above accuracy by evaluating with 5-fold cross validation, reporting the average performance score.
Bonus(es): Find the customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize. Find out What makes the customers buy? - Which feature should be the focuse be on.

# Solution:

The dataset has no duplicates or null values. It comprises 40,000 records with a variety of attributes. Notably, the dataset includes both binary and categorical attributes, providing a diverse set of information for analysis.

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 40000 entries, 0 to 39999
Data columns (total 14 columns):
   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   age        40000 non-null  int64 
 1   job        40000 non-null  object
 2   marital    40000 non-null  object
 3   education  40000 non-null  object
 4   default    40000 non-null  object
 5   balance    40000 non-null  int64 
 6   housing    40000 non-null  object
 7   loan       40000 non-null  object
 8   contact    40000 non-null  object
 9   day        40000 non-null  int64 
 10  month      40000 non-null  object
 11  duration   40000 non-null  int64 
 12  campaign   40000 non-null  int64 
 13  y          40000 non-null  object
dtypes: int64(5), object(9)
memory usage: 4.3+ MB




