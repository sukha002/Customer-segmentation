# Customer-segmentation
### 1. loading of the data

```
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd 
import numpy as np 
```

![My Skills](https://skillicons.dev/icons?i=python)

### 2. Data Acquistion

```
from google.colab import drive
drive.mount('/content/drive')
path_test = "/content/drive/MyDrive/jashan/abisheksudarshan-customer-segmentation/test.csv"
path_train = "/content/drive/MyDrive/jashan/abisheksudarshan-customer-segmentation/train.csv"

df_train = pd.read_csv(path_train)  # issue loading the data into x_test and y test, 
df_test = pd.read_csv(path_test)  # issue loading the data into x_test and y test, 
df_test = df_test.dropna()
df_train = df_train.dropna()

print(df_train.head())
print()
print(df_test.head())

frames = [df_train, df_test]
df = pd.concat(frames)
df
    
```

![image](https://user-images.githubusercontent.com/31208964/213612935-30c56a0b-9427-4268-b11f-5d02b1a869b1.png)


### 3. Data Cleaning
```
df = df.drop('ID', axis=1)
print(df.isnull().values.ravel().sum())
df = df.dropna() 
print(df.isnull().values.ravel().sum())

```
2154 
<br>
0

```
y = df['Segmentation']
df = df.drop('Segmentation', axis=1)
df
```
![image](https://user-images.githubusercontent.com/31208964/213613453-6504c653-ba40-4cc5-bad6-3147570c769d.png)

### Data Labeling

```
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()# Assigning numerical values and storing in another column

df["Gender"] =  labelencoder.fit_transform(df["Gender"] )
df["Ever_Married"] =  labelencoder.fit_transform(df["Ever_Married"] )
df["Graduated"] =  labelencoder.fit_transform(df["Graduated"] )
df["Profession"] =  labelencoder.fit_transform(df["Profession"] )
df["Spending_Score"] =  labelencoder.fit_transform(df["Spending_Score"] )
df
```
![image](https://user-images.githubusercontent.com/31208964/213613612-66a6711b-b455-44a5-80fe-3f8a5137a6da.png)

#### Evaliating the dependent Variable
```
print(y.isnull().values.ravel().sum())
y = y.dropna() 
print(y.isnull().values.ravel().sum())
y =  labelencoder.fit_transform(y )
```

#### Checking the number of observation for both subsets
```
print("length of df ", (len(df)) )
print("length of y ", (len(y)) )
```
<br>
length of df  6665
<br>
length of y  6665

### 4. Splition into Training and Testing
```
from sklearn.model_selection import train_test_split

# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(df,y , random_state=104, test_size=0.25, shuffle=True)
# printing out train and test sets
print('X_train : ')
print(X_train.head())
print('')
print('X_test : ')
print(X_test.head())
print('')
print('y_train : ')
print(y_train)
print('')
print('y_test : ')
print(y_test)  
```
![image](https://user-images.githubusercontent.com/31208964/213614014-4b3e3cd0-41ed-4dc4-9378-b90556d7d594.png)

### 5. Scaling of the Data Frame
```
from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

```

### 6. Keras Modeling
```
model = Sequential()
model.add(Dense(5, input_dim=9, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=2, shuffle=False)   
```
![image](https://user-images.githubusercontent.com/31208964/213614251-396cd7fc-4622-4234-9745-7b23803a810d.png)
![image](https://user-images.githubusercontent.com/31208964/213614304-3e8962e9-801b-4a54-addd-c34f425c9c61.png)
```
import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
![image](https://user-images.githubusercontent.com/31208964/213614479-c2deaae8-04f5-4970-bfe9-7da54cdc6534.png)


     


