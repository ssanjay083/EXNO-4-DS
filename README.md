# EXNO:4-DS
```
       Name:Sanjay s
       Reg No: 212224110047
```
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       import pandas as pd
       from scipy import stats
       import numpy as np
       df=pd.read_csv("bmi.csv")
       df.head()
![image](https://github.com/user-attachments/assets/58e3bef2-ff6a-411e-bea8-83b8df4aa11f)

       df.dropna()
![image](https://github.com/user-attachments/assets/4154e0ec-5205-4d4f-8d75-e8fdee41276d)

       max_vals=np.max(np.abs(df[['Height','Weight']]))
       max_vals
![image](https://github.com/user-attachments/assets/dd31fece-2cf6-430a-a32a-89f4dd070734)

       from sklearn.preprocessing import StandardScaler
       sc=StandardScaler()
       df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
       df.head(10)
![image](https://github.com/user-attachments/assets/091c9b4c-8746-4bcd-aff8-e9f017a18a35)

       from sklearn.preprocessing import MinMaxScaler
       scaler=MinMaxScaler()
       df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
       df.head(10)
![image](https://github.com/user-attachments/assets/648bf075-6e45-4a83-9ef6-47f9d3243dc6)

       from sklearn.preprocessing import Normalizer
       scaler=Normalizer()
       df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
       df
![image](https://github.com/user-attachments/assets/ac101509-1712-4391-9af0-eaa53fd46571)

       df1=pd.read_csv("/content/bmi.csv")
       from sklearn.preprocessing import MaxAbsScaler
       scaler=MaxAbsScaler()
       df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
       df1
![image](https://github.com/user-attachments/assets/d193e6b9-aa88-4a69-a17d-adad2eeeb736)

       df2=pd.read_csv("/content/bmi.csv")
       from sklearn.preprocessing import RobustScaler
       scaler=RobustScaler()
       df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
       df2.head()
![image](https://github.com/user-attachments/assets/d1c34647-e07e-4ad0-b6ac-efba40663058)

FEATURE SELECTION

       import pandas as pd
       import numpy as np
       from scipy.stats import chi2_contingency
       import seaborn as sns
       tips=sns.load_dataset('tips')
       tips.head()
![image](https://github.com/user-attachments/assets/93ccb3b5-1ac4-402c-a610-3eeeae01c989)

       contingency_table=pd.crosstab(tips['sex'],tips['time'])
       print(contingency_table)
![image](https://github.com/user-attachments/assets/a3db4845-2e41-4e8a-b82b-0443f849e566)

       chi2, p, _, _ = chi2_contingency(contingency_table)
       print(f"Chi-Square Statistic: {chi2}")
       print(f"P-value: {p}")
![image](https://github.com/user-attachments/assets/5005a280-33eb-4e21-9dee-402540824806)

       import pandas as pd
       from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
       data={
           'Feature1':[1,2,3,4,5],
           'Feature2': ['A','B','C','A','B'],
           'Feature3':[0,1,1,0,1],
           'Target' :[0,1,1,0,1]
       }
       df=pd.DataFrame(data)
       X=df[['Feature1','Feature3']]
       y=df['Target']
       selector=SelectKBest(score_func=mutual_info_classif, k=1)
       X_new = selector.fit_transform (X,y)
       selected_feature_indices = selector.get_support(indices=True)
       selected_features = X.columns[selected_feature_indices]
       print("Selected Features:")
       print(selected_features)
![image](https://github.com/user-attachments/assets/ab4f4349-906e-4e7c-a698-041535d719df)

# RESULT:
          Hence , the Feature Scaling and Feature Selection process of the dataset has been carried out successfully.
