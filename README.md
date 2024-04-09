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

# let's view the first 5 rows of the data
data.head()
age	job	marital	education	default	balance	housing	loan	contact	day	month	duration	campaign	y
0	58	management	married	tertiary	no	2143	yes	no	unknown	5	may	261	1	no
1	44	technician	single	secondary	no	29	yes	no	unknown	5	may	151	1	no
2	33	entrepreneur	married	secondary	no	2	yes	yes	unknown	5	may	76	1	no
3	47	blue-collar	married	unknown	no	1506	yes	no	unknown	5	may	92	1	no
4	33	unknown	single	unknown	no	1	no	no	unknown	5	may	198	1	no 

