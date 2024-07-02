## What ?

your goal is to compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) you encountered in this section of the program. You will use a dataset related to the marketing of bank products over the telephone.

As part of this exercise, I have examined the dataset from Portuguese banking institution, which has information about a collection of the results of multiple marketing campaigns. Leveraging this data and by applying various classifier models, I have explored the factors that influence the decision to purchase a term deposit product. 


- Here is the [Data](https://github.com/csonamohan/module_17_starter/blob/main/data/bank-additional/bank-additional-full.csv)
- Here is the [Notebook](https://github.com/csonamohan/module_17_starter/blob/main/prompt_III.ipynb)


## Objective

The objective is to to compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) to accurately predict if client will subscribe to a term deposit. By identifying customers who are likely to subcribe, we aim to proactively direct the campaigns to them through targeted marketing and thus increase campaign efficiency and reduce costs involved.

## Target Variable:

The target variable is "y", which has two classes:

"Yes": Customers who has subscribed to term deposit "No": Customers who has not subscribed to term deposit

## Benefits:

Achieving this goal will allow us to:

Increase customer revenue by implementing personalized term deposit acquisition campaigns. Reduce marketing costs by focusing on customers who have a greater propensity to get term deposits. Improve customer satisfaction by addressing customer needs.


## Exploring the data

A detailed look at the data some interesting facts - 

![Alt text](images/1.png)
![Alt text](images/2.png)


## Pre-processing Data - Phase 1

The data provided encompasses 20 attributes, providing valuable insights about the potential customers. I took the following steps to understand that data and prepare it for data modeling - 

- Identified features and target variables
- Found the number of null values in the dataset - there were none.
- Found the datatype of the different features and converted non-numeric features to numeric with the help of one-hot-encoding and LabelBinarizer
- Scaled the data in preparation of modeling
- Ran Principal Component analysis with scaled data. Plotted Explained Variance Ratio vs. Number of Components to reduce dimensions.

![Alt text](images/3.png)

## Modeling Phase 1

After making the necessary changes, I split the dataset into test and train sets. I then ran the different classifier models - Baseline, k-nearest neighbors, logistic regression, decision trees, and support vector machines with default hyper parameters: 

![Alt text](images/4.png)

## Observation:

- Decision tree has the highest Test and training accuracy but training time is high.
- KNN has the next highest Test and training accuracy with shortest training time.
- SVC has the next highest Test and training accuracy but with highest training time.
- Logistic regression has decent training and test accuracy with 3rd highest training time
​

## Pre-processing Data - Phase 2

Feature elimination: We can explore the features with correlation matrix to identify the most irrelevant features and drop them before modeling.

![Alt text](images/5.png)

On re-running PCA with the dataset without the low correlation features, we see an opportunity to reduce dimensions further.

![Alt text](images/6.png)

## Modeling Phase 2 - Optimizing hyper parameters

We can use gridsearchcv to explore the different hyperparameter values for the different classification models and arrive at the best results.

![Alt text](images/7.png)

## Conclusion

It is clear that there are a few features have a greater influence on target variable. Using gridsearchcv, it is possible to make the the different models more accurate with more optimized hyper parameters KNN Classifier, Decisiontree Classifier and SVC show similar test and train acccuracy. However, the training time is the least with KNN and most with SVC.

## Next steps / Recommendations

We would continue with fine tuning the study - 
    
1. Cross-Validation: We could do K-Fold Cross-Validation by splitting the data into k subsets, training the models k times, each time using a different subset as the validation set and the remaining data as the training set.
2. Regularization - We could do L1 and L2 Regularization by adding regularization terms to the loss function to prevent overfitting.
Dropout (for Neural Networks): Randomly drop units during training to prevent co-adaptation.
3. Early Stopping - We could monitor model performance on a validation set and stop training when performance starts to degrade.
4. Better Model Evaluation with different metrics - We could use Precision, Recall, F1 Score, AUC-ROC, AUC-PR, and Specificity instead of just Accuracy.