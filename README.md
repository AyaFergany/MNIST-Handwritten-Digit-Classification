# MNIST-Handwritten-Digit-Classification
Machine Learning Project Using Decision Tree Classifier, Support Vector Classifier, Logistic Regression, Random Forest Classifier

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

