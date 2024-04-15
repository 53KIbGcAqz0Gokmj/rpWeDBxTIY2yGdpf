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
![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/9ebd1eca-4bab-439a-9eeb-7aad590701a7)

# Building and training Models
## Comparing all models till now
![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/2d1928cd-be1a-4ee9-a6d4-8e992708540a)

![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/80087f6c-f68d-4aaa-92fc-ea75e574f546)

# Model comparisons (confusion matrix, ROC, AUC whichever is applicable)
* We are now done with pre-processing and evaluation criterion, so let's start building the model.
# Model Building with original data.

![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/c5514000-35ed-4bd1-8ca2-4f45a1a0704f)

![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/11985d0a-47f4-46bd-b590-87da6673fd33)

## Bagging Classifier

## Training performance:

![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/2a99cd27-7a09-4827-8ed0-443c646ac13d)

* Performance on training set varies between 0.925 to 0.928 recall.
* Let's check the performance on testing data.
  
## Testing performance:
![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/f7bd0aa5-ac2c-43f2-b4a1-e890d2c36e85)

* Performance on testingg set varies between 0.998 to 0.928 recall.

![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/94978a9c-1229-4ca9-aaa8-17c5cf2484db)
   
![download](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/00d72c4b-ab94-48d2-a881-5227d5a84d66)

![image](https://github.com/53KIbGcAqz0Gokmj/rpWeDBxTIY2yGdpf/assets/143815258/b9814352-8ccd-4929-a8da-15d71b75c5e4)

## Conclusion:
The analysis led to the following conclusions:

After evaluating seven different models, it is evident that their performance values significantly vary.

Accuracy: The bagging model achieved the highest accuracy value, followed by the Adaboost model, where as Xgboost, Decision Tree, and Random forest values were found to be overfitting.

The comparative analysis has been carried out by using 5-fold cross-validation methodology and the average score was found to be best for the Bagging Classifier model which was found to be 0.99.

Overfitting: It is important to note that while some models performed well on training data, they exhibited signs of overfitting when tested on the validation or test dataset. This means they may not generalize well to new data.

The Bagging Classifier model showed remarkable robustness, performing consistency well across various test scenarios and datasets, making it a strong contender in real-world deployment.

Business Context: Models that offer interpretability, such as the Bagging classifier model preferred in cases where understanding the model's decision-making process is crucial.

Ultimately, the choice of the best model is a Bagging Classifier and aligning with the given dataset characteristics, and constraints of the project. Further analysis, model fine tuning, or ensemble methods could be considered to improve the model’s performance.










