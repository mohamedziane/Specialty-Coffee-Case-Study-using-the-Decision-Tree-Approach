
# Specialty Coffee Case Study using the Decision Tree Approach

<p align="center">
  <img width="800" height="500" src="https://www.tastingtable.com/img/gallery/coffee-brands-ranked-from-worst-to-best/intro-1640648130.webp">
</p>

## 1. Introduction: Coronavirus

Let's suppose that I have been hired by a rising popular specialty coffee company - RR Diner Coffee - as a data scientist.

RR Diner Coffee sells two types of thing:

- specialty coffee beans, in bulk (by the kilogram only) 
- coffee equipment and merchandise (grinders, brewing equipment, mugs, books, t-shirts).

RR Diner Coffee has three stores, two in Europe and one in the USA. The flagshap store is in the USA, and everything is quality assessed there, before being shipped out. Customers further away from the USA flagship store have higher shipping charges. 

I've been taken on at RR Diner Coffee because the company are turning towards using data science and machine learning to systematically make decisions about which coffee farmers they should strike deals with. 

RR Diner Coffee typically buys coffee from farmers, processes it on site, brings it back to the USA, roasts it, packages it, markets it, and ships it (only in bulk, and after quality assurance) to customers internationally. These customers all own coffee shops in major cities like New York, Paris, London, Hong Kong, Tokyo, and Berlin. 

Now, RR Diner Coffee has a decision about whether to strike a deal with a legendary coffee farm (known as the **Hidden Farm**) in rural China: there are rumors their coffee tastes of lychee and dark chocolate, while also being as sweet as apple juice. 

It's a risky decision, as the deal will be expensive, and the coffee might not be bought by customers. The stakes are high: times are tough, stocks are low, farmers are reverting to old deals with the larger enterprises and the publicity of selling *Hidden Farm* coffee could save the RR Diner Coffee business. 

My first task, then, is ***to build a decision tree to predict how many units of the Hidden Farm Chinese coffee will be purchased by RR Diner Coffee's most loyal customers.*** 

To this end, my team and I have conducted a survey of 710 of the most loyal RR Diner Coffee customers, collecting data on the customers':

- age
- gender 
- salary 
- whether they have bought at least one RR Diner Coffee product online
- their distance from the flagship store in the USA (standardized to a number between 0 and 11) 
- how much they spent on RR Diner Coffee products on the week of the survey 
- how much they spent on RR Diner Coffee products in the month preeding the survey
- the number of RR Diner coffee bean shipments each customer has ordered over the preceding year. 

I also asked each customer participating in the survey whether they would buy the Hidden Farm coffee, and some (but not all) of the customers gave responses to that question. 

I sit back and think: if more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, I will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, I won't strike the deal and the Hidden Farm coffee will remain in legends only. There's some doubt in my mind about whether 70% is a reasonable threshold, but it'll do for the moment. 

To solve the problem, then, I will build a decision tree to implement a classification solution. 

## 2. Objectives

This notebook uses decision trees to determine whether the factors of salary, gender, age, how much money the customer spent last week and during the preceding month on RR Diner Coffee products, how many kilogram coffee bags the customer bought over the last year, whether they have bought at least one RR Diner Coffee product online, and their distance from the flagship store in the USA, could predict whether customers would purchase the Hidden Farm coffee if a deal with its farmers were struck. 

## 3. Dataset loading

- Importing packages
- Loading The Data
- Exploring the data

## 4. Cleaning, Transforming, and Visualizing

- RR coffee customers, in majority (303 vs. 171) , appear to be very loyal as most of them said that they would try the new products.

- For the RR customers who are not willing to try new products, they also appear to be generally spending less compared to the majority based on the spending trend of the week prior to the study.

All in all, the prospect of having a new product appear to please the majority of existing customers.

<p align="center">
  <img width="400" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_boxplot.png">
</p>

<p align="center">
  <img width="400" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_spending.png">
</p>

**Can we admissibly conclude anything from this scatterplot?** 

- The customers who answered "Yes" appear to spend more when they are closer to the flasgship store in the US. The reverse trend is notable for the customers who answered "No".

- There is an imbalance between people who voted "Yes" (303) vs. those who answered "No" (171) which may potentially affect the models score accuracy later on.

## 5. Modelling

**Model 1: Entropy model - no max_depth: Interpretation and evaluation**

<p align="center">
  <img width="800" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_model1.png">
</p>


- We have 9 leaves with purity at 0 (This is the ideal scenario), however 4 of them are having a very low number of samples so we cannot be very confident for their predictions (clear sign of over-fitting).

- Features used in the splits: spend_last_month, Distance and Age (features importance illustrated below).

- We have a fully grown Decision Tree (none of the parameters were set e.g., max_depth) as a result the tree grows to a full  depth of 5.

- Not limiting the growth of a Decision Tree will delay reaching the split choices that will get us to the pure nodes (leaves=predictions) causing over-fitting.

**Model 2: Gini impurity model - no max_depth**

Gini impurity, like entropy, is a measure of how well a given feature (and threshold) splits the data into categories.

Their equations are similar, but Gini impurity doesn't require logorathmic functions, which can be computationally expensive.

<p align="center">
  <img width="800" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_model2.png">
</p>

- We have 11 leaves with purity at 0 (This is the ideal scenario), however 6 of them are having a very low number of samples so we cannot be very confident for their predictions (clear sign of over-fitting).

- Features used in the splits: spend_last_month, Distance, Age and num_coffeeBags_per_year (features importance illustrated below).

- We have a fully grown Decision Tree (none of the parameters were set e.g., max_depth) as a result the tree grows to a full  depth of 6.

- Not limiting the growth of a Decision Tree will delay reaching the split choices that will get us to the pure nodes (leaves=predictions) causing over-fitting.

**Model 3: Entropy model - max depth 3**

We're going to try to limit the depth of our decision tree, using entropy first.  

As you know, we need to strike a balance with tree depth. 

Insufficiently deep, and we're not giving the tree the opportunity to spot the right patterns in the training data.

Excessively deep, and we're probably going to make a tree that overfits to the training data, at the cost of very high error on the (hitherto unseen) test data. 

Sophisticated data scientists use methods like random search with cross-validation to systematically find a good depth for their tree. We'll start with picking 3, and see how that goes. 

<p align="center">
  <img width="600" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_model3.png">
</p>

- We have 2 leaves with purity at 0 (One Class) with only one leave having a very low number of samples (This is the ideal scenario): The more leaves with purity=0 and high samples the more the information gain (more predicition power).

- Features used in the splits: spend_last_month & Distance (features importance illustrated below).

- We have a Decision Tree (max_depth=3) as a result the tree grows to a depth of 3. 

- When limiting the growth of the Decision Tree we find the split choices that will get us to the pure nodes much faster resulting in more information gain (more prediction power).

**Model 4: Gini impurity  model - max depth 3**
We're now going to try the same with the Gini impurity model. 

<p align="center">
  <img width="600" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_model4.png">
</p>

- We have 5 leaves with purity at 0  with only one leave having a very low number of samples (This is the ideal scenario): The more leaves with purity=0 and high samples the more the information gain (more predicition power).

- Features used in the splits: spend_last_month and Distance (features importance illustrated below).

- We have a Decision Tree (max_depth=3) as a result the tree grows to a depth of 3.  When limiting the growth of the Decision Tree we find the split choices that will get us to the pure nodes much faster resulting in more information gain (more predicition power).

## 6. Random Forest

You might have noticed an important fact about decision trees. Each time we run a given decision tree algorithm to make a prediction (such as whether customers will buy the Hidden Farm coffee) we will actually get a slightly different result. This might seem weird, but it has a simple explanation: machine learning algorithms are by definition ***stochastic***, in that their output is at least partly determined by randomness. 

To account for this variability and ensure that we get the most accurate prediction, we might want to actually make lots of decision trees, and get a value that captures the centre or average of the outputs of those trees. Luckily, there's a method for this, known as the ***Random Forest***. 

Essentially, Random Forest involves making lots of trees with similar properties, and then performing summary statistics on the outputs of those trees to reach that central value. Random forests are hugely powerful classifers, and they can improve predictive accuracy and control over-fitting. 

Why not try to inform your decision with random forest? You'll need to make use of the RandomForestClassifier function within the sklearn.ensemble module, found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

<p align="center">
  <img width="600" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_model5.png">
</p>

## 7. Conclusion

* Random Forest Model 5 has shown an increase of the potential buyers by 3.2% from 183 to 189 potetial buyers and that put us exactly at 70.1 % which is more than indicated 70% of the interviewed customers who are likely to buy the Hidden Farm coffee.

- According to the above we can encourage RR Diner Coffee to strike the deal with the local Hidden Farm farmers.

- Why Random Forests outerperform Gini Model2, even with much lower Accuracy?

- Because our Dataset is imbalanced; accuracy should not be used to measure our Models Performance.

- Instead of Accuracy, we need to focus on the Confusion Matrix that shows the correct predictions and types of incorrect predictions, as shown below the prediction power of "Yes" has increased in Random Forest compared to the Gini Model.

- In addition to the confusion matrix we can use ROC AUC from sklearn.metrics to evaluate metrics for calculating the performance of any classification model's performance, as shown below AUC score is signficantly higher in Random Forest compared to the Gini Model

**Gini impurity model - max depth 3:**

- Actual Data  distributed as "NO"=41 and "YES"=78
- TP (True Positives): Model predicted "NO" and it was actually "NO" = 39
- TN (True Negatives): Model predicted "YES" and it was actually "YES" = 77
- FN (False Negatives): Model predicted "YES" and it was actually "No" = 2

**Random Forest model 5 - max depth 3:**
- Actual Data  distributed as "NO"=41 and "YES"=78
- TP (True Positives): Model predicted "NO" and it was actually "NO" = 35 (Less than Gini Model)
- TN (True Negatives): Model predicted "YES" and it was actually "YES" = 77 (Similar to Gini)
- FN (False Negatives): Model predicted "YES" and it was actually "No" = 6 (Increased by 4 compared to Gini)

<p align="center">
  <img width="600" height="400" src="https://raw.githubusercontent.com/mohamedziane/Specialty-Coffee-Case-Study-using-the-Decision-Tree-Approach/main/images/img_finalcompmetrics.png">
</p>
