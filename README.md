# MNIST-Handwritten-Digit-Classification
Machine Learning Project Using Decision Tree Classifier, Support Vector Classifier, Logistic Regression, Random Forest Classifier

# Simple introduction to logistic regression
Before we dive into understanding logistic regression, let us start with some basics about the different types of machine learning algorithms.

 What are the differences between supervised learning, unsupervised learning & reinforcement learning?
Machine learning algorithms are broadly classified into three categories - supervised learning, unsupervised learning, and reinforcement learning.

Supervised Learning - Learning where data is labeled and the motivation is to classify something or predict a value. Example: Detecting fraudulent transactions from a list of credit card transactions.

Unsupervised Learning - Learning where data is not labeled and the motivation is to find patterns in given data. In this case, you are asking the machine learning model to process the data from which you can then draw conclusions. Example: Customer segmentation based on spend data.

Reinforcement Learning - Learning by trial and error. This is the closest to how humans learn. The motivation is to find optimal policy of how to act in a given environment. The machine learning model examines all possible actions, makes a policy that maximizes benefit, and implements the policy(trial). If there are errors from the initial policy, apply reinforcements back into the algorithm and continue to do this until you reach the optimal policy. Example: Personalized recommendations on streaming platforms like YouTube.

What are the two types of supervised learning?

As supervised learning is used to classify something or predict a value, naturally there are two types of algorithms for supervised learning - classification models and regression models.

Classification model - In simple terms, a classification model predicts possible outcomes. Example: Predicting if a transaction is fraud or not.
Regression model - Are used to predict a numerical value. Example: Predicting the sale price of a house.

# What is logistic regression?
Logistic regression is an example of supervised learning. It is used to calculate or predict the probability of a binary (yes/no) event occurring. An example of logistic regression could be applying machine learning to determine if a person is likely to be infected with COVID-19 or not. Since we have two possible outcomes to this question - yes they are infected, or no they are not infected - this is called binary classification.

In this imaginary example, the probability of a person being infected with COVID-19 could be based on the viral load and the symptoms and the presence of antibodies, etc. Viral load, symptoms, and antibodies would be our factors (Independent Variables), which would influence our outcome (Dependent Variable).

How is logistic regression different from linear regression?
In linear regression, the outcome is continuous and can be any possible value. However in the case of logistic regression, the predicted outcome is discrete and restricted to a limited number of values.

For example, say we are trying to apply machine learning to the sale of a house. If we are trying to predict the sale price based on the size, year built, and number of stories we would use linear regression, as linear regression can predict a sale price of any possible value. If we are using those same factors to predict if the house sells or not, we would logistic regression as the possible outcomes here are restricted to yes or no.

Hence, linear regression is an example of a regression model and logistic regression is an example of a classification model.

# Introduction for Support Vector Classifier
Support vector machine is highly preferred by many as it produces significant accuracy with less computation power. Support Vector Machine, abbreviated as SVM can be used for both regression and classification tasks. But, it is widely used in classification objectives.

What is Support Vector Machine?
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space
(N — the number of features) that distinctly classifies the data points.

![image](https://user-images.githubusercontent.com/91394241/217568365-cc70806a-ad5a-42a9-b6ab-49ab0736a692.png)![image](https://user-images.githubusercontent.com/91394241/217568422-ab47771b-2c36-4a8f-a66a-a9ae689af29f.png)

                                     Possible hyperplanes
                                  
To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

Hyperplanes and Support Vectors

![image](https://user-images.githubusercontent.com/91394241/217568473-1066a5b8-435c-446e-9e2a-719673e3230f.png)

                           Hyperplanes in 2D and 3D feature space


Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. It becomes difficult to imagine when the number of features exceeds 3.


# Introduction for Decision Tree classifier

Decision tree learning is a supervised learning approach used in statistics, data mining and machine learning. In this formalism, a classification or regression decision tree is used as a predictive model to draw conclusions about a set of observations.

Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. More generally, the concept of regression tree can be extended to any kind of object equipped with pairwise dissimilarities such as categorical sequences.

Decision trees are among the most popular machine learning algorithms given their intelligibility and simplicity.

In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. In data mining, a decision tree describes data (but the resulting classification tree can be an input for decision making).

# Introduction for Random Forest Classifier

Working of Random Forest Algorithm


![image](https://user-images.githubusercontent.com/91394241/217578326-6c0062cb-55e8-4487-a9fa-9d377170ad45.png)

                           IMAGE COURTESY: javapoint

The following steps explain the working Random Forest Algorithm:

Step 1: Select random samples from a given data or training set.

Step 2: This algorithm will construct a decision tree for every training data.

Step 3: Voting will take place by averaging the decision tree.

Step 4: Finally, select the most voted prediction result as the final prediction result.

This combination of multiple models is called Ensemble. Ensemble uses two methods:

Bagging: Creating a different training subset from sample training data with replacement is called Bagging. The final output is based on majority voting. 
Boosting: Combing weak learners into strong learners by creating sequential models such that the final model has the highest accuracy is called Boosting. Example: ADA BOOST, XG BOOST. 

![image](https://user-images.githubusercontent.com/91394241/217578457-6af85ca9-5a62-481b-b369-69831228ebcb.png)


Bagging: From the principle mentioned above, we can understand Random forest uses the Bagging code. Now, let us understand this concept in detail. Bagging is also known as Bootstrap Aggregation used by random forest. The process begins with any original random data. After arranging, it is organised into samples known as Bootstrap Sample. This process is known as Bootstrapping.Further, the models are trained individually, yielding different results known as Aggregation. In the last step, all the results are combined, and the generated output is based on majority voting. This step is known as Bagging and is done using an Ensemble Classifier.

![image](https://user-images.githubusercontent.com/91394241/217578552-aeba14e4-0d88-4074-b5d8-8a82d8498472.png)








# Sources 
https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

https://en.wikipedia.org/wiki/Decision_tree_learning

https://www.simplilearn.com/tutorials/machine-learning-tutorial/random-forest-algorithm

https://www.capitalone.com/tech/machine-learning/what-is-logistic-regression/
