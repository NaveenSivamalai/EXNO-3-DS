## EXNO-3-DS
# NAME:NAVEEN S
# REG NO:212222110030
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
# OUTPUT:
<img width="1273" height="437" alt="Screenshot 2025-10-09 133416" src="https://github.com/user-attachments/assets/0963531e-19ab-4eb7-93bf-0daacd062698" />

```
# ORDINAL ENCODING
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
# OUTPUT:
<img width="1223" height="306" alt="Screenshot 2025-10-09 133426" src="https://github.com/user-attachments/assets/aaa4adc7-fea1-4182-ac15-fe61876b5471" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
# OUTPUT:
<img width="1193" height="402" alt="Screenshot 2025-10-09 133433" src="https://github.com/user-attachments/assets/3f2552b6-83e9-4453-8a05-9c88a69bba74" />

```
# Label Encoder ( orders in alphabetical order)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
# OUTPUT:
<img width="1197" height="484" alt="Screenshot 2025-10-09 133443" src="https://github.com/user-attachments/assets/1942e084-663b-4d74-80b1-3e8805e0a2f2" />

```
# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```
# OUTPUT:
<img width="1201" height="510" alt="Screenshot 2025-10-09 133449" src="https://github.com/user-attachments/assets/e7900e8e-4ca1-49aa-8fb7-f7ad879047c2" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
# OUTPUT:
<img width="1210" height="401" alt="Screenshot 2025-10-09 133457" src="https://github.com/user-attachments/assets/298862b6-6fd5-42fd-a862-dff621b97375" />

```
pip install --upgrade category_encoders
```
# OUTPUT:
<img width="1178" height="560" alt="Screenshot 2025-10-09 133505" src="https://github.com/user-attachments/assets/3a85104a-78dc-4b3d-b141-d22e82ae9420" />

```
# BINARY ENCODER
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
# OUTPUT:
<img width="1186" height="447" alt="Screenshot 2025-10-09 133705" src="https://github.com/user-attachments/assets/9096d47d-908a-494d-b5b0-5fdba7435b4b" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
# OUTPUT:
<img width="1207" height="455" alt="Screenshot 2025-10-09 133711" src="https://github.com/user-attachments/assets/a3485073-015b-453a-b851-939f1e016da4" />

```
# MEAN ENCODING
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
# OUTPUT:
<img width="1199" height="501" alt="Screenshot 2025-10-09 133719" src="https://github.com/user-attachments/assets/27778bcb-09c6-402e-b967-dfa7d7d29529" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
# OUTPUT:
<img width="1203" height="531" alt="Screenshot 2025-10-09 133726" src="https://github.com/user-attachments/assets/c7c3baba-7b76-4eff-9edb-c8241e47c6d4" />

```
df.skew()
```
# OUTPUT:
<img width="1205" height="158" alt="Screenshot 2025-10-09 133734" src="https://github.com/user-attachments/assets/fd89835a-94ad-4046-983d-b28b5518ec92" />

```
# 1. LOG TRANSFORMATION
np.log(df["Highly Positive Skew"])
```
# OUTPUT:
<img width="1209" height="318" alt="Screenshot 2025-10-09 133740" src="https://github.com/user-attachments/assets/af2a2e00-e0ed-4e3f-9a26-adc241a62f52" />

```
# 2. RECIPROCAL TRANSFORMATION
np.reciprocal(df["Moderate Positive Skew"])
```
# OUTPUT:
<img width="1212" height="297" alt="Screenshot 2025-10-09 133746" src="https://github.com/user-attachments/assets/bfc90745-0d28-4add-87d5-65299dbcb2b3" />

```
# 4. SQUARE ROOT TRANSFORMATION
np.sqrt(df["Highly Positive Skew"])
```
# OUTPUT:
<img width="1208" height="343" alt="Screenshot 2025-10-09 133753" src="https://github.com/user-attachments/assets/cda77f91-666f-4b6b-a07e-acc038d2fae8" />

```
# 5. SQUARE TRANSFORMATION
np.square(df["Highly Positive Skew"])
```
# OUTPUT:
<img width="1212" height="328" alt="Screenshot 2025-10-09 133759" src="https://github.com/user-attachments/assets/07b4ff50-faa2-4f12-b074-de018053b015" />

```
# POWER TRANSFORMATIONS
# BOX COX
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
# OUTPUT:
<img width="1208" height="502" alt="Screenshot 2025-10-09 133807" src="https://github.com/user-attachments/assets/b99667e0-b001-4d2c-ad7e-c52988b822ca" />

```
df.skew()
```
# OUTPUT:
<img width="1208" height="184" alt="Screenshot 2025-10-09 133814" src="https://github.com/user-attachments/assets/5cdb73f7-af07-44f9-8ebe-6814fabee15d" />

```
# YEO_JOHNSON
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
# OUTPUT:
<img width="1206" height="245" alt="Screenshot 2025-10-09 133821" src="https://github.com/user-attachments/assets/e0467fa5-c7d1-4a46-bbb8-9018e3d33e31" />

```
# QUANTILE TRANSFORMATION
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
# OUTPUT:
<img width="1206" height="553" alt="Screenshot 2025-10-09 133828" src="https://github.com/user-attachments/assets/6656b0f9-3733-46bb-809f-d3f99425bc99" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
# OUTPUT
<img width="1217" height="606" alt="Screenshot 2025-10-09 133834" src="https://github.com/user-attachments/assets/b082963a-01a1-4579-96d5-1c32ff1b55bd" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
# OUTPUT:
<img width="1221" height="590" alt="Screenshot 2025-10-09 133839" src="https://github.com/user-attachments/assets/fce80824-460b-4817-b7a4-aaf519e82ed1" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
# OUTPUT:
<img width="1217" height="595" alt="Screenshot 2025-10-09 133845" src="https://github.com/user-attachments/assets/9dd6e68d-e2d8-4315-8b42-6e060eede5d8" />

# RESULT:
Thus the program executed successfully.

       
