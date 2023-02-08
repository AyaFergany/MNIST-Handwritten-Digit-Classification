# MNIST-Handwritten-Digit-Classification
Machine Learning Project Using Decision Tree Classifier, Support Vector Classifier, Logistic Regression, Random Forest Classifier

# Introduction
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
