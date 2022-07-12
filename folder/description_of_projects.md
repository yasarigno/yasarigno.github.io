## A Big Data project on AWS to assist a very young AgriTech start-up, called "Fruits!".

The goal is to provide to the general public a mobile application that would allow users to take a picture of a fruit and obtain information about this fruit. My mission was to implement a first version of the fruit image classification engine and to build in a Big Data environment a first data processing chain which will include preprocessing and a dimensionality reduction step. For this, I have used Amazon Web Services (AWS) cloud tools to build the first data processing chain. 

Technical tools used in this project are AWS (S3, EC2, IAM), DataBricks, PySpark, Transfer Learning via ResNet50, Principal Component Analysis, Computer Vision, Comparison of Big Data Architectures.

In this project, I applied the following techniques/strategies
 - Strategy adopted to distribution of calculations between the machines
 - Extraction of features of images via Transfer Learning
 - Dimensionality reduction
 - Amazon Web Services (IAM, EC2, S3)
 - DataBricks integration/configuration through AWS
 - Object-oriented programming vs Functional programming

🔗 Link GitHub 

https://github.com/yasarigno/Projet_8

📝 Data source:

https://www.kaggle.com/moltean/fruits

## Banks Credit Risk Prediction with Machine Learning.
I implemented Machine Learning algorithms to predict whether a client of HOME CREDIT will be able to repay a loan or he/she will have repayment difficulties. The company Home Credit wants to implement a “credit scoring” tool to calculate the probability that a customer will repay their credit, then classify the request as granted or refused credit. Therefore, they wish to develop a classification algorithm based on various data sources (behavioural data, data from other financial institutions, etc.).

Technical tools used in this project are Scikit-Learn, Linear Regression, KNN, Gradient Boosting, Random Forest, Correlation heat maps, Pair plots, Confusion matrix, XGBoost, Heroku, Streamlit, Resampling the data : oversampling, undersampling, SMOTE, Random under-sampling.

In this project, I applied the following techniques/strategies
 - Feature Engineering, Encoding Categorical Variables (OneHotEncoder, Label Encoding)
 - Handling missing values, detecting anomalies, outliers
 - Handling highly unbalanced datasets
 - Visualization via kernel density estimation plot (KDE) 
 - Aligning Training and Testing Data
 - Standardisation via MinMaxScaler
 - Testing different Machine Learning algorithms 
 - Model interpretation, Feature importance
 - Creation of a dashboard

🔗 Link GitHub 

https://github.com/yasarigno/Projet_7

📝 Data source:

https://www.kaggle.com/c/home-credit-default-risk/data

## Recognition of categories of products from images and textual descriptions.

I was given a dataset which consists of pictures of goods and their descriptions. The CSV file contains some other information such as the price of the product, the name of the product and its brand. There is also a variable "product_category_tree" which defines categories and 6 subcategories of the product. This variable is defined manually by the sellers. As the size of our dataset grows up drastically, the task of associating the product to the category will be a burden. Therefore we must automate this task by using only the pictures and the descriptions. Now the problem that we want to solve is converted into a problem of Natural Language Processing (NLP) and that of Computer Vision (CV). We approach this problem of recognition of categories from different aspects. First we use only the descriptions and perform algorithms of NLP, then in later notebooks we take the tools of CV into account.

Technical tools used in this project are Pandas, Numpy. On textual data: TF-IDF, NLTK, Spacy, GloVe, Gensim (Word2Vec), LDA; On visual data: OpenCV (SIFT, ORB), Convolutional Neural Networks, VGG-16, Tensorflow, PCA, t-SNE.

In this project, I applied the following techniques/strategies
 - Text mining, text categorization, text clustering
 - Operations on corpus (stemming, lemmatisation, bag-of-words, removing stopwords, visualization via word clouds etc.)
 - Word embeddings with different dimensions via GloVe
 - Image processing (contrast, grayscales, filters, size etc.)
 - Vectorization 
 - Data augmentation 
 - Transfer learning via ResNet50 and VGG-16
 - Clustering via KNN
 - Visualization and dimensionality reduction via t-SNE
 - Testing different strategies of Computer Vision

🔗 Link GitHub 

https://github.com/yasarigno/Projet_6

📝 Data source:

https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+prétraitement+textes+images.zip

## Classification of customer profiles of OLIST, an e-commerce site.

I provide OLIST's marketing team with customer segmentation that they can use on a daily basis for their communication campaigns. The aim is to provide the marketing team a description of customer segmentation and its underlying logic for optimal use. In this project we analyse types of customers through their behaviour and personal data. We therefore use unsupervised methods to group customers with similar profiles. The customers have been classified into 8 profiles. We also provide OLIST a maintenance contract proposal based on an analysis of the stability.

Technical tools used in this project are KNN, DBSCAN, Hierarchical clustering, Elbow method, Scree plot, PCA, t-SNE, Correlation circles, PEP8

In this project, I applied the following techniques/strategies
 - Merging several datasets
 - Data cleaning and exploratory data analysis
 - Feature engineering
 - Clustering, classification via unsupervised learning
 - Metrics for clustering (silhouette score, ARI etc.)
 - Interpretation of clusters (radar plot, heatmap, parallel coordinates plot)
 - Maintenance of the final model

🔗 Link GitHub 

https://github.com/yasarigno/Projet_5

📝 Data source:

https://www.kaggle.com/olistbr/brazilian-ecommerce

## Predictions of energy consumption of the city of Seattle in order to reach its goal of being a carbon neutral city in 2050.

The City of Seattle is looking at the carbon emissions and energy use of non-residential properties by 2050. We will make predictions of
carbon dioxide (CO2) emissions and
total energy consumption of the city of Seattle
from existing data. Our mission is to analyse past data, which was carried out in 2015 and 2016, and to implement Machine Learning algorithms to predict future values that have not yet been measured. Since readings are expensive to obtain, our study was carried out without annual consumption readings. We will first carry out a short exploratory analysis, test different prediction models in order to best respond to the problem and assess the impact of the EnergyStarScore indicator on the performance of these models.

Technical tools used in this project are Pandas, Numpy, Seaborn, Matplotlib, Scikit-Learn, Folium.

Techniques used in the project are
 - Predictive and exploratory analysis 
 - Handling data leakage
 - Data cleaning (detection of anomalies, outliers, imputation of missing values)
 - Metrics (RMSE, R2 etc.)
 - Testing ML models (Linear Regression, Lasso, Ridge, Elastic Net, SVM, K Neighbors Regressor, Random Forest Regressor, Decision Tree Regressor, Gradient Boosting Regressor)
 - Optimisation of the best final model



## Building a mobile application for Santé Publique France. 
The Santé publique France has launched a call for projects to find innovative ideas for food-related mobile applications. We propose NUTRI + Z that will provide the user: NutriScore, which is a letter grade varying from A (best) to E (worst), an alert Z which inform the user whether the product likely have adverse effects on health or on the environment. We are working on a dataset which consist of foods or beverages. The alert Z takes into account the material of the packaging, the ingredients of the product. For instance, that its package contains plastics will mean that the product may pollute the environment. Therefore, the user will be informed about the negative effects of consumption of that product. We are responsible for verifying if the database of OpenFoodFacts contains enough information to implement the application project. In other words, we provide a result on the feasibility of our project.

Technical tools used in this project are Pandas, Numpy, Math, SciPy, Seaborn, Matplotlib, WordCloud, Knn Imputer.

In this project, I applied the following techniques/strategies

- Data cleaning on structured datasets
- Analysis of the database openfoodfacts.org
- Handling missing values
- Visualisation by pie charts
- Communicating results using readable and relevant graphic representations (box plot, pair plot, QQ plot etc.)
- Correlation Matrix, Linear Correlation Coefficient, ANOVA
- Categorical variables, multivariate statistical analysis


