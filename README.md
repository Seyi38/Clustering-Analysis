# Clustering-Analysis
The objective of this task is to provide insights into the socioeconomic and health factors of the countries in the dataset.

# DATA UNDERSTANDING:

# Data Structure:

The data provided contains eight socio-economic and health indices of 167 countries. 

# Exploratory Data Analysis:

The data was read into a Pandas dataframe. Quantitative analysis and visualization of the data was carried out using various python packages such as seaborn, matplotlib etc.

Descriptive statistics of the features in the dataset present varying information of the data that requires more analysis. For example, with a standard deviation of 40.3 the child mortality presents values that are on average close to the mean child mortality rate of 38.27. This suggests that most of the countries in the dataset have a child mortality rate with a little over two points away from the mean. This was not commonplace among the other features.
As can be seen in the fig 2.0, income and gross domestic product per person accounts for the largest values in the dataset. 

 ![image](https://user-images.githubusercontent.com/20461822/218774196-cce0596b-9b17-40a2-af9b-7a4388c2916f.png)

	Fig 2.0: Socioeconomic and health features

The distribution of these features alongside the export and import is skewed with the median not centered. This skewness may suggest a few countries are responsible for large amounts of these features.
A look at the relationship between the features show less independence among themselves.
Exports, health, income present positive linear relationships with domestic product per person (GDPP). It is also observed that child mortality is high for countries with low GDPP. This is seen in fig 2.1.

 ![image](https://user-images.githubusercontent.com/20461822/218774313-754fe1ac-22f5-43dd-a3aa-094674029d4f.png)

	Fig 2.1: relationship between features

In other to generate more insights and understand how these countries can be categorized based on their socio-economic and health indices, an unsupervised machine learning algorithm called clustering is used.

# DATA PREPARATION

# Data Cleaning:

The following steps were taken to ensure the data was clean before proceeding to modeling.
1.	Checking all data were in the right format
2.	A check for missing/null values was done with no null/missing values in any of the columns.
3.	A check for duplicates was carried out with no duplicate values found.

Feature(s) selection/Justification:
1.	The relationship between features was explored and two sets of features were selected for the different models.

# MODEL BUILDING:

An unsupervised machine learning algorithm known as K-means clustering is used as the data had no labels. The idea of clustering is to partition the data into groups known as clusters based on common properties among the features.

# K-means clustering:
K-means is a clustering algorithm that randomly selects an initial cluster centroid. The model repetitively computes the cluster centers until an optimal centroid is found for the number of clusters. In K-means, the number of clusters is specified before fitting the model with the data.

To build the clustering model, the following steps were taken:
1.	Rescaled the features to have a mean of 0 and standard deviation of 1 by a process called standardization. This allows the algorithm to assign importance to all features.
2.	Compute silhouette coefficient and within clusters sum of squares to select the best k value.
3.	Initialize model with a k means value and fit the data
4.	Make predictions on the data

Two models were built starting with a small number of features and a second model with all features.

For the first model, income, child mortality and life expectancy were the selected features. 

To select the number of clusters, a range of clusters from 1-10 were computed and their silhouette coefficient and WCSS values observed. This is shown in fig 2.2 and 2.3.

# MODEL EVALUATION:
To evaluate the model, the following metrics were calculated based on the K number of clusters: 
1.	Silhouette coefficient: The silhouette coefficient, a measure of the similarities and dissimilarities within clusters is used. Here, how well the model separates dissimilar observations and groups similar observations are taken into consideration. Values range from -1 to 1 with 1 indicating high density clusters and -1 indicating wrong cluster assignment. Scores close to zero is an indication for overlapping clusters
2.	Elbow method: In this method, the within cluster sum of square known as WCSS (i.e sum of squared distance between each point and the centroid in a cluster) for each value of K is computed. The WCSS vs the K value is plotted and as K increases, WCSS reduces. The point at which there is a little reduction in WCSS as K increases forms an elbow shape and is the optimal K value.

The following plots for the silhouette coefficient and elbow method were obtained for model 1 and 2. 

Model 1 with small number of features:

 ![image](https://user-images.githubusercontent.com/20461822/218774436-55e50be9-240e-4830-94c4-dc6e26295148.png)

	Fig 2.2(a): Plot of showing elbow method

 ![image](https://user-images.githubusercontent.com/20461822/218774479-4256f0fa-abf4-4450-92b8-26f4b19d73d6.png)

	Fig 2.2(b): Plot of showing silhouette coefficient


Model 2 with all features selected:

![image](https://user-images.githubusercontent.com/20461822/218774567-aa160b60-7186-4600-a280-236e20aabcb2.png)

 	Fig 2.3(a): Plot of showing elbow method
![image](https://user-images.githubusercontent.com/20461822/218774668-f4c57630-2abd-4650-9ec9-d830bff0340f.png)
	Fig 2.3(b): Plot of showing silhouette coefficient


For both models, 3 clusters were selected.

Increasing the number of features led to a slightly lower silhouette coefficient for the same K number of clusters (3). Increasing the clusters in model 2 to 4 does not offer any significant improvement in the model coefficient.
From the experimentation a shift in the cluster centers between the two models are observed despite retaining the same number of clusters. This is seen the fig 2.4a and fig 2.4b. 
  ![image](https://user-images.githubusercontent.com/20461822/218774744-744ceda7-6629-41b1-961d-af731ff9f1ba.png)
  
	Fig 2.4	: Cluster Centers for Model 1		Fig 2.4b: Cluster Centers for Model 2

# ANALYSIS OF CLUSTERS

The groupings by the model are seen by the average values for the socio-economic and health factors. As fig 2.5 shows, Income, GDPP, life expectancy is highest for cluster 0 while lowest for cluster 2 with cluster 1 in between. The reverse is the case for factors such as child mortality and inflation as seen in the charts below.
![image](https://user-images.githubusercontent.com/20461822/218774820-2b4af530-6cc6-4eae-86b2-5c4571eb243e.png)

     Fig 2.5a: Average socio-economic factors for clusters

![image](https://user-images.githubusercontent.com/20461822/218775146-4a81c1b3-5534-4c48-8555-5498640f96c7.png) ![image](https://user-images.githubusercontent.com/20461822/218775193-46019f7d-42d6-4b11-8857-784eba5846ea.png)


  	 Fig 2.5b: Average socio-economic factors for clusters

Based on the properties of each identified clusters, they can be categorized into underdeveloped, developing and developed countries. The boxplot in fig 2.6a,b,c showing the distribution of figures for export, import and income confirms the earlier suspicion that a select group/cluster of countries are responsible for a larger share of the figures. 
![image](https://user-images.githubusercontent.com/20461822/218775301-072c9497-c8fb-473f-ac14-c60a464f8d6b.png) ![image](https://user-images.githubusercontent.com/20461822/218775328-e128534f-12a1-47d0-bb4b-5e4cf1ea0303.png)

 	Fig 2.6a: Income per clusters 	Fig 2.6b: Imports per clusters
 ![image](https://user-images.githubusercontent.com/20461822/218775432-c9001344-d9ab-4b75-a93b-65ede632db45.png)
	
	Fig 2.6c: Exports per clusters

These select group of countries responsible are categorized as developed countries. Fig 2.7a shows the countries with highest income. 

 	![image](https://user-images.githubusercontent.com/20461822/218775494-498695da-23f8-4e30-b9ec-1684e80325ad.png) ![image](https://user-images.githubusercontent.com/20461822/218775516-3a086f44-053d-4743-9202-b23d0394e029.png)

	 Fig 2.7a: High income countries	Fig 2.7b: Countries with lowest inflation

Countries with highest child mortality are all grouped in the same clusters. In fig 2.7b we observe that Seychelles, Slovenia, Latvia, Bahamas, and Lebanon despite belonging to a different cluster group (developing clusters) have very low inflation values. 


# CONCLUSION/RECOMMENDATION

From the analysis, different categories for countries have been effectively found and have provided more insights into the general socioeconomic and health factors of the different countries. This makes intervention design and policies easier as they can be targeted at groups instead of individual countries reducing, cost, wastage, and manpower. The relationship between income and negative factors like child mortality and life expectancy are seen. For countries in the underdeveloped categories, policies and interventions that will boost economic activities such as manufacturing and trade volume are recommended as this will lead to an increase in income and GDPP. It is also recommended that some level of assistance is rendered to countries in the developing category to enable increased productivity leading to higher income and GDPP. This model may be improved using dimension reduction and feature extraction algorithms like PCA before fitting the K-means.

