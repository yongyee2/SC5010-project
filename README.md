# SC5010-project
# SC5010-project
### Contributors
  - Guan Jia Sheng (U2040851A)
  - Nguyen Ngoc Minh Truc (U1940862C)
  - Lim Yong Yee (U2040881F)

### 4 notebooks:
### 1. EDA
  - Identify pattern within dataset
  - Check association between variables and disorders
### 2. Missing Value Treatment & Machine Learning 
  - Iterative imputation
  - Model family selection
  - Model optimisation and building
  - Model evaluation 
  - Based on full dataset without assumptions
### 3. Replaced Values 
  - Alternative chained model 
  - Built with dataset values replaced based on assumptions obtained from EDA
### 4. Extra
  - Fill missing values of predictors with mode of each column 
  - Comparison of missing value treatment between iterative imputation and mode
  
## Problem Definitions
 - Predict classes (Genetic Disorder) and subclasses (Disorder Subclass) in the Genetic Disorder dataset
  - Identify the best model for prediction of classes and subclasses
  - To identify the most important variables for prediction

## Exploratory Data Analysis
  - The analysis is found within the EDA notebook.

## Missing Values Treatment
  - Genetic Disorder column directly related to the Disorder Subclass
  - Missing values in Genetic Disorder directly filled with values based on the Disorder Subclass
Diabetes![missing value](https://user-images.githubusercontent.com/92092401/163960654-0e2e6bd6-7bf8-45ec-a797-e6044bc11d54.JPG)



  - Missing values in other predictors - Tried two methods 
    - Simply filling in with the mode of the predictor column (SimpleImputer)
    - Iterative imputation of missing values (IterativeImputer) 
  - IterativeImputer uses the multivariate imputation by chained equations approach
    - Each predictor is modelled as a function of other predictors and the missing values were predicted imputed
    - Naïve Bayes model was used in iterative imputation as label encoding was done initially just for imputation purpose and Naïve Bayes can ignore ordinality of data
    - Process repeats until the maximum number of iterations decided is reached or convergence occurs whereby there are hardly any more changes in values as compared to the previous iteration 
  - IterativeImputer and SimpleImputer yielded similarity in this dataset when a Random Forest model is built
  - Either one can be used in this dataset

![simpleimputer](https://user-images.githubusercontent.com/92092401/163961066-2e836b43-3a35-49ae-b2a1-fb0d09a679d5.JPG)

### Imbalance Treatment
Imbalance
  - Performance of models created were evaluated using original imbalance data
  - The results serve as a control to compare with other imbalance treatment methods
  - The imbalanced dataset was hypothesised to produce the worse performing models

### Undersampling
  - Majority classes, from Genetic Disorder or Genetic Subclass, were undersampled randomly without replacement 
  - The sampling strategy was set to use “not minority” which resamples all classes except the minority class
  - Shorter computational time for model training as dataset size was reduced
  - May risk removing important data points that should have been used for model training

### Oversampling
  - Minority classes, from Genetic Disorder or Genetic Subclass, were oversampled by picking samples at random with replacement 
  - The sampling strategy was defaulted to use “not majority” which resample all classes but the majority class
  - Avoid the removal of important data points as done in undersampling
  - Computational runtime was significantly increased

### SMOTE
  - Instead of oversampling minority class by duplication, SMOTE synthesize new examples from the minority class which are close in the feature space. 
  - Provide additional information to the model
  - Sampling strategy was defaulted to use “not majority” which resample all classes but the majority class. 
  - Best imbalance treatment method amongst the ones tried.

### Comparison of imbalance treatment methods
![imbalance treatment](https://user-images.githubusercontent.com/92092401/163961302-801c2e13-3c5b-4457-a5be-1d178f927409.JPG)

  - Class imbalance was noticeable in both Genetic Disorder and Disorder Subclass
  - The performance of imbalance treatment decreases in the following order: SMOTE > Oversampling > Undersampling > Imbalance 
  - Error bars are standard deviation.

## Models Tested
### Supervised Learning
### Support Vector Machine
  - SVM breaks the multiclass problem into multiple binary classification problems through the one-to-one approach 
  - Finds a hyperplane that best separates between every two classes, and is optimised by maximising the margin
  - Large genetic disorder dataset with about 20000 rows meant that training time using SVM was high so it was not used as the final model. 
  - SVM Training time complexity = O(n3) VS Random forest Training time complexity = O(n*log(n)*d*k)
    - where n = number of training samples, k = number of decision trees, d = dimensionality of the data.
  - SVM was not carried out for SMOTE and Oversampling treatment on dataset with train test split stratified against Disorder Subclass due to immense runtime

### Naive Bayes
  - Naive Bayes is a probabilistic model which can be used in Multiclass Classification, suitable for prediction of Genetic Disorder and Disorder Subclass. 

![eqn 1](https://user-images.githubusercontent.com/92092401/163961444-2152a46c-371d-4799-80d1-b4fb655c24ca.JPG)

  - Bayes Theorem allows the finding of probability of a Genetic Disorder class (hypothesis) happening given a certain set of predictor values (evidence)
  - Assumption of Bayes Theorem
    - Predictors are independent and that all predictors have equal effect on the outcome 
  - Due to the denominator being a constant, it can be removed in the above equation to give the equation below.
![eqn 2](https://user-images.githubusercontent.com/92092401/163961524-134a9509-2913-468a-a22c-a13bd49858f0.JPG)

  - Class of Genetic Disorder assigned by finding the class with the highest probability with the given predictors
![eqn 3](https://user-images.githubusercontent.com/92092401/163961597-82485ecc-1005-4b2b-af6a-fc12e393d912.JPG)

  - Naive Bayes did not work as well as other supervised learning models in this project and thus was not chosen to be used in the final model

### Random Forest
  - Random Forest utilises the ensemble technique and is suitable for a Multiclass problem. 
  - By creating a bunch of decision trees that use different variables and data points for training, collaborative learning can be achieved and the class assigned ultimately will be via a “vote” whereby the class will be the one that is predicted by most decision trees. 
  - Best performing model out of all supervised learning models and was used for the final models.

### AdaBoost
  - Boosting algorithm aims to improve the prediction power by converting a number of weak learners to strong learners
  - Decision trees with 1 levels were used as the weak models - Decision stumps
  - Model mechanics
    - A model was first built and equal weights were given to all the data points
    - Higher weights were then assigned to points that were wrongly classified
    - Points with higher weight were then given more importance in the subsequent model as they will be oversampled within the new dataset picked
    - The process will continue until the error is minimised. 
  - Perform better than Naive Bayes in general but underperformed in comparison to Random Forest and Support Vector Machine. 


### Supervised Learning Model Performance Comparison - SMOTE and Stratified against Genetic Disorder
![11](https://user-images.githubusercontent.com/92092401/163964357-c3e7ec9d-070f-4c39-8881-e35c61ede311.JPG)
![12](https://user-images.githubusercontent.com/92092401/163964994-43578f97-4412-4012-80ba-eec4d884ea81.JPG)
  - Support Vector Machine had the highest accuracy for train and test set but was overfitted
  - Random Forest performed relatively consistently throughout for accuracy and cross validation results
  - Random Forest was decided to be the final model for Genetic Disorder model and Disorder Subclass models
  - AdaBoost was performing better than Naive Bayes most of the time but was not chosen due to poorer results when compared to Random Forest
  - Error bars are standard deviation

### Random Forest Model Performance Comparison after SMOTE - Stratified against Genetic Disorder and Disorder Subclass
![13](https://user-images.githubusercontent.com/92092401/163966110-7a33c310-eccc-453e-807b-f671f6f9e70a.JPG)
![14](https://user-images.githubusercontent.com/92092401/163966211-56191c98-e0de-4952-b2ee-2f07f70796b4.JPG)


  - Genetic Disorder model
  - Stratification against Genetic Disorder during train test split showed better performance than stratification against Disorder Subclass
  - Numbers of Alzheimer’s and Cancer cases are extremely small compared to total number of observations, stratification against Disorder Subclass was done to avoid excluding these data points from the train or test sets for both main and submodels (to keep same train and test set for all models)
  - If prediction of Genetic Disorder classes was the only problem, then stratification against Genetic Disorder would give much better results. 
  - Error bars are standard deviation


### Unsupervised Learning
### KModes

  - As the predictors were mainly categorical variables to begin with and the few continuous variables were all converted to categorical variables, KMeans would no longer be an option to be used for clustering. KModes was used instead. 
  - Works by calculating dissimilarity between the data points. 
  - The optimal number of clusters, also the hyperparameter K, was determined by plotting an Elbow curve and the optimal number was chosen at the area where the curve bends. 
  - 6 was chosen for all KModes performed after different imbalance treatment methods.  Therefore, 6 points were picked at random to label as clusters. 
  - Dissimilarities were then calculated and each observation was assigned to the closest cluster. New modes of clusters were redefined and the process repeats until there is no more re-assignment of points.
  - After clustering with all types of imbalance treatments, all were found to be ineffective in grouping the different Genetic Disorder classes, particularly between single-gene and mitochondrial disorders.
![kmode](https://user-images.githubusercontent.com/92092401/163967565-2c3dc442-6e93-444d-8872-bdc130eb39a7.JPG)


### Anomaly Detection - KNN

  - Used the K-Nearest Neighbour model
  - Anomalies of the dataset were extracted out after one-hot encoding
  - Hyperparameter was set such that a data point distance to its kth nearest neighbour is viewed as the outlying score and it can be interpreted as a measure of density
  - The contamination was set to change from 0.01 to 0.1
  - Multifactorial genetic inheritance disorders were less likely to be flagged as outliers at low contamination rate



## Final Models 

  - Main model was chained to 3 sub models
  - Prediction of Genetic Disorder value by the main model decides which submodel to use to predict the Disorder Subclass value
  - Model to use was decided based on comparison of performance between different models




  - Built the final model using SMOTE imbalance treatment and stratifying against Disorder Subclass 
    - Main model had accuracy of 62.7 and 63.3 percent for the train and test set respectively, above the baseline of 33.3 percent
    - Combined model had accuracy of 24.9 and 24.3 percent for the train and test set respectively, above the baseline of 11.1 percent
  - Error bars are standard deviation.





  - Variable importance was similar for Genetic Disorder model and Disorder Subclass models. 
  - Only the genetic inheritance factors and symptoms were found to be important predictors.
  - Due to the important role of genes in the manifestation of diseases. Given that different diseases can have different symptoms, it is also essential in the identification of disease.
  - Most of the variables turned out to be useless as predicted from the exploratory data analysis. 
  - Therefore, only useful predictors were kept to create models that are lower in complexity. 
  - The mean and standard deviation of accumulation of the impurity decrease for each variable within each tree of the Random Forest models was compared visually and plotted as illustrated above. 
  - Error bars are standard deviation.



## Conclusion

  - Symptoms and genes are not named in the original dataset description
  - Difficult to associate the exact meaning of each symptoms and genes with any of the disorders, or explain why the frequency of the symptom and gene variables deemed important for prediction was similar across all 3 disorders 
  - Genetic Disorder and the Disease Subclass were heavily impacted by the patients’ genes 
    - Expected from a biological standpoint 
  - Disease may manifest different symptoms
    - Make sense that symptoms will shed light on both Genetic Disorder and the Disease Subclass.

## References
  - https://machinelearningmastery.com/iterative-imputation-for-missing-values-in-machine-learning/
  - https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
  - https://www.youtube.com/watch?v=peh2l4dePBc
  - https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/
  - https://towardsdatascience.com/introducing-anomaly-outlier-detection-in-python-with-pyod-40afcccee9ff
  - https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/
  - https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html?highlight=undersample#imblearn.under_sampling.RandomUnderSampler
  - https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html?highlight=oversample#imblearn.over_sampling.RandomOverSampler
  - https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html?highlight=smote#imblearn.over_sampling.SMOTE
  - https://7-hiddenlayers.com/time-complexities-of-ml-algorithms/
