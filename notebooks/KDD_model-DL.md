
# Data pre process


```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, RandomizedSearchCV,GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.backend as K
from keras.optimizers import SGD, Adam 
from keras.models import load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


import warnings
warnings.filterwarnings("ignore")
```


```python
dt_path = "../data/kdd2009/"
```


```python
#first 190 features are numerical and the last 40 are categorical
X = pd.read_table(dt_path+"orange_small_train.data")
X.shape
```




    (50000, 230)




```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
```


```python
#X = reduce_mem_usage(X)
```

## features transformation


```python
#X = reduce_mem_usage(X)
```


```python
num_features = list(X.columns[:190])
cat_features = list(X.columns[190:230])
```


```python
empty_features = []
c = 0.3
for feat in X.columns:
    nulls = X[feat].isnull().value_counts()
    try:
        not_nulls = nulls[False]
        if not_nulls < c*40000:
            empty_features.append(feat)
    except:    
        empty_features.append(feat)
print ("number of empty features is", len(empty_features))
```

    number of empty features is 154



```python
#remove sparse features
for feat in empty_features:
    #data.drop(feat, axis = 1, inplace = True)    
    if feat in num_features:
        num_features.remove(feat)
    else:
        cat_features.remove(feat)
    X.drop(feat,axis=1, inplace=True)
```

## missing value imputation
X[num_features].fillna(0, axis=1, inplace=True)

```python
#Numeric features: replace missing values with (max value + 1).
maxs = X.max(axis = 0)
for i,feat in enumerate(num_features):
    fill_value = maxs[i] + 1.
    X.fillna({feat: fill_value}, inplace=True)
```


```python
#Categorial features: replace values with their frequencies.
data_cat_all = X[cat_features]

for feat in data_cat_all.columns:
    data_cat_all[feat] = data_cat_all[feat].map(data_cat_all.groupby(feat).size())
    
X[cat_features] = data_cat_all.loc[:39999,:]
```


```python
#Categorial features: replace missing values with zeros.
X.fillna(0., inplace=True)
```
X[cat_features] = X[cat_features].apply(lambda x:pd.factorize(x)[0])X[cat_features]=X[cat_features].astype('category')
## Load target valuable


```python
y = pd.read_table(dt_path+"orange_small_train_churn.labels",header=None)
y.shape
```




    (50000, 1)




```python
y = LabelEncoder().fit_transform(y)
```


```python
pd.Series(y).value_counts() 
```




    0    46328
    1     3672
    dtype: int64



## get data ready for training


```python
Scaler = MinMaxScaler()
X = pd.DataFrame(Scaler.fit_transform(X),columns=X.columns,index=X.index)
```


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2018, stratify=y)
```

# Modeling

## Single Sequential Model


```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.backend as K
from keras.optimizers import SGD, Adam 
from keras.models import load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

```


```python
NB_EPOCH = 200
BATCH_SIZE = 4096
VERBOSE = 1
OPTIMIZER = 'adam'
```


```python
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1],kernel_initializer='random_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(16, input_dim=X_train.shape[1],kernel_initializer='random_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(8, kernel_initializer='random_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1,kernel_initializer='random_uniform'))
model.add(Activation('sigmoid'))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_21 (Dense)             (None, 32)                2464      
    _________________________________________________________________
    activation_21 (Activation)   (None, 32)                0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 32)                0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 16)                528       
    _________________________________________________________________
    activation_22 (Activation)   (None, 16)                0         
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 16)                0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 8)                 136       
    _________________________________________________________________
    activation_23 (Activation)   (None, 8)                 0         
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 8)                 0         
    _________________________________________________________________
    dense_24 (Dense)             (None, 1)                 9         
    _________________________________________________________________
    activation_24 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 3,137
    Trainable params: 3,137
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer=OPTIMIZER,
              loss='binary_crossentropy',
              metrics=['accuracy'])
```


```python
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
```


```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=20, verbose=0, mode='auto'),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]
```


```python
model.fit(np.array(X_train), np.array(y_train), epochs=NB_EPOCH, batch_size=BATCH_SIZE,callbacks=callbacks)
```

    Epoch 1/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2567 - acc: 0.9266
    Epoch 2/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2577 - acc: 0.9266
    Epoch 3/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2582 - acc: 0.9266
    Epoch 4/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2559 - acc: 0.9265
    Epoch 5/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2554 - acc: 0.9265
    Epoch 6/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2570 - acc: 0.9265
    Epoch 7/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2562 - acc: 0.9266
    Epoch 8/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2560 - acc: 0.9265
    Epoch 9/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2551 - acc: 0.9266
    Epoch 10/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2553 - acc: 0.9265
    Epoch 11/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2547 - acc: 0.9265
    Epoch 12/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2556 - acc: 0.9265
    Epoch 13/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2569 - acc: 0.9265
    Epoch 14/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2569 - acc: 0.9265
    Epoch 15/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2563 - acc: 0.9266
    Epoch 16/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2573 - acc: 0.9266
    Epoch 17/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2563 - acc: 0.9265
    Epoch 18/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2553 - acc: 0.9265
    Epoch 19/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2556 - acc: 0.9265
    Epoch 20/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2557 - acc: 0.9265
    Epoch 21/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2554 - acc: 0.9265
    Epoch 22/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2553 - acc: 0.9266
    Epoch 23/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2557 - acc: 0.9265
    Epoch 24/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2560 - acc: 0.9265
    Epoch 25/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2551 - acc: 0.9266
    Epoch 26/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2558 - acc: 0.9265
    Epoch 27/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2557 - acc: 0.9266
    Epoch 28/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2543 - acc: 0.9265
    Epoch 29/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2537 - acc: 0.9265
    Epoch 30/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2551 - acc: 0.9265
    Epoch 31/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2558 - acc: 0.9265
    Epoch 32/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2547 - acc: 0.9265
    Epoch 33/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2559 - acc: 0.9266
    Epoch 34/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2557 - acc: 0.9265
    Epoch 35/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2557 - acc: 0.9266
    Epoch 36/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2552 - acc: 0.9266
    Epoch 37/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2532 - acc: 0.9265
    Epoch 38/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2549 - acc: 0.9265
    Epoch 39/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2556 - acc: 0.9266
    Epoch 40/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2562 - acc: 0.9266
    Epoch 41/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2540 - acc: 0.9265
    Epoch 42/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2550 - acc: 0.9266
    Epoch 43/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2534 - acc: 0.9265
    Epoch 44/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2546 - acc: 0.9265
    Epoch 45/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2546 - acc: 0.9265
    Epoch 46/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2546 - acc: 0.9265
    Epoch 47/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2550 - acc: 0.9265
    Epoch 48/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2548 - acc: 0.9265
    Epoch 49/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2534 - acc: 0.9265
    Epoch 50/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2543 - acc: 0.9265
    Epoch 51/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2538 - acc: 0.9265
    Epoch 52/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2531 - acc: 0.9265
    Epoch 53/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2531 - acc: 0.9266
    Epoch 54/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2542 - acc: 0.9266
    Epoch 55/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2535 - acc: 0.9265
    Epoch 56/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2534 - acc: 0.9266
    Epoch 57/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2527 - acc: 0.9265
    Epoch 58/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2529 - acc: 0.9266
    Epoch 59/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2536 - acc: 0.9265
    Epoch 60/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2526 - acc: 0.9266
    Epoch 61/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2546 - acc: 0.9266
    Epoch 62/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2545 - acc: 0.9266
    Epoch 63/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2537 - acc: 0.9265
    Epoch 64/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2529 - acc: 0.9266
    Epoch 65/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2524 - acc: 0.9266
    Epoch 66/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2532 - acc: 0.9265
    Epoch 67/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2525 - acc: 0.9266
    Epoch 68/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2535 - acc: 0.9266
    Epoch 69/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2536 - acc: 0.9265
    Epoch 70/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2533 - acc: 0.9265
    Epoch 71/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2533 - acc: 0.9266
    Epoch 72/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2527 - acc: 0.9266
    Epoch 73/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2538 - acc: 0.9266
    Epoch 74/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2526 - acc: 0.9266
    Epoch 75/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2534 - acc: 0.9265
    Epoch 76/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2523 - acc: 0.9266
    Epoch 77/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2521 - acc: 0.9265
    Epoch 78/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2528 - acc: 0.9265
    Epoch 79/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2529 - acc: 0.9266
    Epoch 80/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2528 - acc: 0.9265
    Epoch 81/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2526 - acc: 0.9266
    Epoch 82/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2523 - acc: 0.9265
    Epoch 83/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2534 - acc: 0.9265
    Epoch 84/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2528 - acc: 0.9266
    Epoch 85/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2529 - acc: 0.9265
    Epoch 86/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2516 - acc: 0.9265
    Epoch 87/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2520 - acc: 0.9265
    Epoch 88/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2515 - acc: 0.9265
    Epoch 89/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2540 - acc: 0.9265
    Epoch 90/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2532 - acc: 0.9266
    Epoch 91/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2513 - acc: 0.9265
    Epoch 92/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2521 - acc: 0.9266
    Epoch 93/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2533 - acc: 0.9265
    Epoch 94/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2524 - acc: 0.9266
    Epoch 95/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2531 - acc: 0.9265
    Epoch 96/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2510 - acc: 0.9265
    Epoch 97/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2516 - acc: 0.9265
    Epoch 98/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2518 - acc: 0.9266
    Epoch 99/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2510 - acc: 0.9266
    Epoch 100/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2520 - acc: 0.9266
    Epoch 101/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2518 - acc: 0.9266
    Epoch 102/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2493 - acc: 0.9265
    Epoch 103/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2510 - acc: 0.9265
    Epoch 104/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2527 - acc: 0.9265
    Epoch 105/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2519 - acc: 0.9266
    Epoch 106/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2522 - acc: 0.9265
    Epoch 107/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2512 - acc: 0.9265
    Epoch 108/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2507 - acc: 0.9266
    Epoch 109/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2515 - acc: 0.9265
    Epoch 110/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2510 - acc: 0.9266
    Epoch 111/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2513 - acc: 0.9265
    Epoch 112/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2509 - acc: 0.9266
    Epoch 113/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2495 - acc: 0.9266
    Epoch 114/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2500 - acc: 0.9266
    Epoch 115/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2519 - acc: 0.9266
    Epoch 116/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2499 - acc: 0.9265
    Epoch 117/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2501 - acc: 0.9265
    Epoch 118/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2514 - acc: 0.9266
    Epoch 119/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2498 - acc: 0.9265
    Epoch 120/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2505 - acc: 0.9266
    Epoch 121/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2500 - acc: 0.9266
    Epoch 122/200
    40000/40000 [==============================] - 0s 3us/step - loss: 0.2502 - acc: 0.9266





    <keras.callbacks.History at 0x7fcc3c4bc9b0>




```python
# training performance
y_pred_proba = model.predict_proba(np.array(X_test), batch_size=512)
```


```python
roc_auc_score(y_test, y_pred_proba)
```




    0.7070505042900976




```python
flattened = [val for sublist in y_pred_proba for val in sublist]
pd.Series(flattened).describe()
```

## SkLearn Wrapper


```python
def dl_models(optimizer='rmsprop', init='glorot_uniform'):
    #create model
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1],kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, input_dim=X_train.shape[1],kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='random_uniform'))
    model.add(Activation('sigmoid'))
    #compile model
    model.compile(optimizer=OPTIMIZER,
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## Cross Validation


```python
NB_EPOCH = 200
BATCH_SIZE = 4096
OPTIMIZER = 'adam'
K_FOLD = 10
LAMBD = 0.002 #0.000005
SEED = 12#np.random.randn()

kfold = StratifiedKFold(n_splits=K_FOLD, shuffle=False, random_state=SEED)
models = []
cvscores_training = []
cvscores_validaton = []
ind = 0.4

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=15, epsilon=0.0001, min_lr=0.0000001)
earlyStopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=40, verbose=0, mode='auto')

callbacks = [reduce_lr,earlyStopping]

for train, test in kfold.split(X, y):
    
    model = Sequential()
    model.add(Dense(100, input_dim=X.shape[1],kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed = SEED))
    model.add(Dense(50, kernel_initializer='random_uniform',kernel_regularizer=regularizers.l2(LAMBD)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3, seed = SEED))
    model.add(Dense(16, kernel_initializer='random_uniform',kernel_regularizer=regularizers.l2(LAMBD)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2, seed = SEED))
    model.add(Dense(8, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.1, seed = SEED))
    model.add(Dense(1,kernel_initializer='random_uniform'))
    model.add(Activation('sigmoid'))
    #model.summary()  
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
    
    print('\n model {} starts to fit ... \n'.format(ind))
    model.fit(np.array(X.iloc[train]), np.array(y[train]), epochs=NB_EPOCH, batch_size=BATCH_SIZE, validation_split=0, callbacks=callbacks)
    
    # evaluate the model    
    p_training = model.predict_proba(np.array(X.iloc[train]), batch_size=4096, verbose=0)
    score_training = roc_auc_score(y[train],p_training)
    p_validaton = model.predict_proba(np.array(X.iloc[test]), batch_size=4096, verbose=0)
    score_validaton = roc_auc_score(y[test],p_validaton)
    print("\n AUC: training is {}; validation is {}".format(score_training,score_validaton))
        
    cvscores_training.append(score_training)
    cvscores_validaton.append(score_validaton)
    
#    model_path = '../../models/DL_V2.0/' + 'CV_model_' + str(ind)
#    model.save(model_path)
#    models.append(model_path)
    
    ind += 1
    
print("training AUC: maximum: {}; minimun: {}; mean: {}; sd: {}".format \
      (np.max(cvscores_training), np.min(cvscores_training), np.mean(cvscores_training), np.std(cvscores_training)))
print("validaton AUC: maximum: {}; minimun: {}; mean: {}; sd: {}".format \
      (np.max(cvscores_validaton), np.min(cvscores_validaton), np.mean(cvscores_validaton), np.std(cvscores_validaton)))
```

    
     model 0.4 starts to fit ... 
    
    Epoch 1/200
    44999/44999 [==============================] - 1s 15us/step - loss: 0.6978 - acc: 0.9260
    Epoch 2/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.6873 - acc: 0.9266
    Epoch 3/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.6733 - acc: 0.9266
    Epoch 4/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.6145 - acc: 0.9266
    Epoch 5/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.4142 - acc: 0.9266
    Epoch 6/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3363 - acc: 0.9266
    Epoch 7/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3193 - acc: 0.9266
    Epoch 8/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3101 - acc: 0.9266
    Epoch 9/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3026 - acc: 0.9266
    Epoch 10/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2980 - acc: 0.9266
    Epoch 11/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2942 - acc: 0.9266
    Epoch 12/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2899 - acc: 0.9266
    Epoch 13/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2861 - acc: 0.9266
    Epoch 14/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2822 - acc: 0.9266
    Epoch 15/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2803 - acc: 0.9266
    Epoch 16/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2784 - acc: 0.9266
    Epoch 17/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2753 - acc: 0.9266
    Epoch 18/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2753 - acc: 0.9266
    Epoch 19/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2730 - acc: 0.9266
    Epoch 20/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2722 - acc: 0.9266
    Epoch 21/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2702 - acc: 0.9266
    Epoch 22/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2697 - acc: 0.9266
    Epoch 23/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2691 - acc: 0.9266
    Epoch 24/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2668 - acc: 0.9266
    Epoch 25/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2673 - acc: 0.9266
    Epoch 26/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2664 - acc: 0.9266
    Epoch 27/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2652 - acc: 0.9266
    Epoch 28/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2653 - acc: 0.9266
    Epoch 29/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2638 - acc: 0.9266
    Epoch 30/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2637 - acc: 0.9266
    Epoch 31/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2624 - acc: 0.9266
    Epoch 32/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2616 - acc: 0.9266
    Epoch 33/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2614 - acc: 0.9266
    Epoch 34/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2611 - acc: 0.9266
    Epoch 35/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2596 - acc: 0.9266
    Epoch 36/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2597 - acc: 0.9266
    Epoch 37/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2596 - acc: 0.9266
    Epoch 38/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2577 - acc: 0.9266
    Epoch 39/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2588 - acc: 0.9266
    Epoch 40/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2589 - acc: 0.9266
    Epoch 41/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2578 - acc: 0.9266
    Epoch 42/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2578 - acc: 0.9266
    Epoch 43/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2574 - acc: 0.9266
    Epoch 44/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 45/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2568 - acc: 0.9266
    Epoch 46/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2563 - acc: 0.9266
    Epoch 47/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2564 - acc: 0.9266
    Epoch 48/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 49/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 50/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2560 - acc: 0.9266
    Epoch 51/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 52/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 53/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 54/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 55/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 56/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 57/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 58/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2544 - acc: 0.9266
    Epoch 59/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 60/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 61/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 62/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 63/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 64/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 65/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 66/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 67/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2533 - acc: 0.9266
    Epoch 68/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 69/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 70/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 71/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 72/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 73/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 74/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 75/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 76/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 77/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 78/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 79/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 80/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 81/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 82/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 83/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 84/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 85/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 86/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 87/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 88/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 89/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 90/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 91/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 92/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 93/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 94/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 95/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 96/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 97/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 98/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 99/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 100/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 101/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 102/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 103/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 104/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 105/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 106/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 107/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 108/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 109/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 110/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 111/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 112/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 113/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 114/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 115/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 116/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 117/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 118/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 119/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 120/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 121/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 122/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 123/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 124/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 125/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 126/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 127/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 128/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 129/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 130/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 131/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 132/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 133/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 134/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 135/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 136/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 137/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 138/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 139/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 140/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 141/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 142/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 143/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 144/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 145/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 146/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 147/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 148/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 149/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 150/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 151/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 152/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 153/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 154/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 155/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 156/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 157/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 158/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 159/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 160/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 161/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 162/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 163/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 164/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 165/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 166/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 167/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 168/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 169/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 170/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 171/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 172/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 173/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 174/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 175/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 176/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 177/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 178/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 179/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 180/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 181/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 182/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 183/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 184/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 185/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 186/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 187/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 188/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 189/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 190/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 191/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 192/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 193/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 194/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 195/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 196/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 197/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 198/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 199/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 200/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    
     AUC: training is 0.7018025515046862; validation is 0.6910338404076615
    
     model 1.4 starts to fit ... 
    
    Epoch 1/200
    44999/44999 [==============================] - 1s 19us/step - loss: 0.6979 - acc: 0.9128
    Epoch 2/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.6872 - acc: 0.9266
    Epoch 3/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.6712 - acc: 0.9266
    Epoch 4/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.5950 - acc: 0.9266
    Epoch 5/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3820 - acc: 0.9266
    Epoch 6/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3385 - acc: 0.9266
    Epoch 7/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3152 - acc: 0.9266
    Epoch 8/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3100 - acc: 0.9266
    Epoch 9/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.3027 - acc: 0.9266
    Epoch 10/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2960 - acc: 0.9266
    Epoch 11/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2929 - acc: 0.9266
    Epoch 12/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2887 - acc: 0.9266
    Epoch 13/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2854 - acc: 0.9266
    Epoch 14/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2835 - acc: 0.9266
    Epoch 15/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2809 - acc: 0.9266
    Epoch 16/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2797 - acc: 0.9266
    Epoch 17/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2774 - acc: 0.9266
    Epoch 18/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2767 - acc: 0.9266
    Epoch 19/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2754 - acc: 0.9266
    Epoch 20/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2735 - acc: 0.9266
    Epoch 21/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2734 - acc: 0.9266
    Epoch 22/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2724 - acc: 0.9266
    Epoch 23/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2710 - acc: 0.9266
    Epoch 24/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2708 - acc: 0.9266
    Epoch 25/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2694 - acc: 0.9266
    Epoch 26/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2689 - acc: 0.9266
    Epoch 27/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2675 - acc: 0.9266
    Epoch 28/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 29/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 30/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2671 - acc: 0.9266
    Epoch 31/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2653 - acc: 0.9266
    Epoch 32/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2651 - acc: 0.9266
    Epoch 33/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2644 - acc: 0.9266
    Epoch 34/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2643 - acc: 0.9266
    Epoch 35/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2628 - acc: 0.9266
    Epoch 36/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2627 - acc: 0.9266
    Epoch 37/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2627 - acc: 0.9266
    Epoch 38/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2621 - acc: 0.9266
    Epoch 39/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2614 - acc: 0.9266
    Epoch 40/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2616 - acc: 0.9266
    Epoch 41/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2607 - acc: 0.9266
    Epoch 42/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2587 - acc: 0.9266
    Epoch 43/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2604 - acc: 0.9266
    Epoch 44/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2589 - acc: 0.9266
    Epoch 45/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2606 - acc: 0.9266
    Epoch 46/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2590 - acc: 0.9266
    Epoch 47/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2590 - acc: 0.9266
    Epoch 48/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2585 - acc: 0.9266
    Epoch 49/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2583 - acc: 0.9266
    Epoch 50/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2577 - acc: 0.9266
    Epoch 51/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2584 - acc: 0.9266
    Epoch 52/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2581 - acc: 0.9266
    Epoch 53/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2580 - acc: 0.9266
    Epoch 54/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2569 - acc: 0.9266
    Epoch 55/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2574 - acc: 0.9266
    Epoch 56/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2570 - acc: 0.9266
    Epoch 57/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2555 - acc: 0.9266
    Epoch 58/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2566 - acc: 0.9266
    Epoch 59/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2566 - acc: 0.9266
    Epoch 60/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2563 - acc: 0.9266
    Epoch 61/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2559 - acc: 0.9266
    Epoch 62/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2556 - acc: 0.9266
    Epoch 63/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2556 - acc: 0.9266
    Epoch 64/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2551 - acc: 0.9266
    Epoch 65/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 66/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 67/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 68/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 69/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 70/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 71/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 72/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2548 - acc: 0.9266
    Epoch 73/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2544 - acc: 0.9266
    Epoch 74/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 75/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 76/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 77/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 78/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 79/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2544 - acc: 0.9266
    Epoch 80/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2540 - acc: 0.9266
    Epoch 81/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 82/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 83/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 84/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 85/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 86/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 87/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 88/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 89/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 90/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 91/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 92/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 93/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 94/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 95/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 96/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 97/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 98/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 99/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 100/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 101/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 102/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 103/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 104/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 105/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 106/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 107/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 108/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 109/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 110/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 111/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 112/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 113/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 114/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 115/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 116/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 117/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 118/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 119/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 120/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 121/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 122/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 123/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 124/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 125/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 126/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 127/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 128/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 129/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 130/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 131/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 132/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 133/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 134/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 135/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 136/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 137/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 138/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 139/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 140/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 141/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 142/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 143/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 144/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 145/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 146/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 147/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 148/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 149/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 150/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 151/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 152/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 153/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 154/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 155/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 156/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 157/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 158/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 159/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 160/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 161/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 162/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 163/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 164/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 165/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 166/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 167/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 168/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 169/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 170/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 171/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 172/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 173/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 174/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 175/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 176/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 177/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 178/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 179/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 180/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 181/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 182/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 183/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 184/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 185/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 186/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 187/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 188/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 189/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 190/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 191/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 192/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 193/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 194/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 195/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 196/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 197/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 198/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 199/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 200/200
    44999/44999 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    
     AUC: training is 0.701753956220182; validation is 0.7011192156457926
    
     model 2.4 starts to fit ... 
    
    Epoch 1/200
    45000/45000 [==============================] - 1s 17us/step - loss: 0.6976 - acc: 0.9266
    Epoch 2/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6866 - acc: 0.9266
    Epoch 3/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6675 - acc: 0.9266
    Epoch 4/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.5735 - acc: 0.9266
    Epoch 5/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3596 - acc: 0.9266
    Epoch 6/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3370 - acc: 0.9266
    Epoch 7/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3151 - acc: 0.9266
    Epoch 8/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3059 - acc: 0.9266
    Epoch 9/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2998 - acc: 0.9266
    Epoch 10/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2943 - acc: 0.9266
    Epoch 11/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2883 - acc: 0.9266
    Epoch 12/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2858 - acc: 0.9266
    Epoch 13/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2840 - acc: 0.9266
    Epoch 14/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2813 - acc: 0.9266
    Epoch 15/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2789 - acc: 0.9266
    Epoch 16/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2790 - acc: 0.9266
    Epoch 17/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2763 - acc: 0.9266
    Epoch 18/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2758 - acc: 0.9266
    Epoch 19/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2747 - acc: 0.9266
    Epoch 20/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2727 - acc: 0.9266
    Epoch 21/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2728 - acc: 0.9266
    Epoch 22/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2713 - acc: 0.9266
    Epoch 23/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2698 - acc: 0.9266
    Epoch 24/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2694 - acc: 0.9266
    Epoch 25/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2693 - acc: 0.9266
    Epoch 26/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2685 - acc: 0.9266
    Epoch 27/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2673 - acc: 0.9266
    Epoch 28/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2676 - acc: 0.9266
    Epoch 29/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2668 - acc: 0.9266
    Epoch 30/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2644 - acc: 0.9266
    Epoch 31/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2651 - acc: 0.9266
    Epoch 32/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2644 - acc: 0.9266
    Epoch 33/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2655 - acc: 0.9266
    Epoch 34/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2640 - acc: 0.9266
    Epoch 35/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2632 - acc: 0.9266
    Epoch 36/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2622 - acc: 0.9266
    Epoch 37/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2634 - acc: 0.9266
    Epoch 38/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2628 - acc: 0.9266
    Epoch 39/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2615 - acc: 0.9266
    Epoch 40/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2616 - acc: 0.9266
    Epoch 41/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2615 - acc: 0.9266
    Epoch 42/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2595 - acc: 0.9266
    Epoch 43/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2600 - acc: 0.9266
    Epoch 44/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2601 - acc: 0.9266
    Epoch 45/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2589 - acc: 0.9266
    Epoch 46/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2582 - acc: 0.9266
    Epoch 47/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2593 - acc: 0.9266
    Epoch 48/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2593 - acc: 0.9266
    Epoch 49/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2585 - acc: 0.9266
    Epoch 50/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2586 - acc: 0.9266
    Epoch 51/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2578 - acc: 0.9266
    Epoch 52/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2579 - acc: 0.9266
    Epoch 53/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2575 - acc: 0.9266
    Epoch 54/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2563 - acc: 0.9266
    Epoch 55/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2569 - acc: 0.9266
    Epoch 56/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2571 - acc: 0.9266
    Epoch 57/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2565 - acc: 0.9266
    Epoch 58/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2565 - acc: 0.9266
    Epoch 59/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2562 - acc: 0.9266
    Epoch 60/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2561 - acc: 0.9266
    Epoch 61/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 62/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2562 - acc: 0.9266
    Epoch 63/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 64/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2556 - acc: 0.9266
    Epoch 65/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2560 - acc: 0.9266
    Epoch 66/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 67/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2561 - acc: 0.9266
    Epoch 68/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 69/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2556 - acc: 0.9266
    Epoch 70/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 71/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 72/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 73/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 74/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 75/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 76/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 77/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2544 - acc: 0.9266
    Epoch 78/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 79/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 80/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2540 - acc: 0.9266
    Epoch 81/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 82/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2540 - acc: 0.9266
    Epoch 83/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 84/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 85/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 86/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 87/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 88/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 89/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 90/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 91/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2533 - acc: 0.9266
    Epoch 92/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 93/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 94/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2539 - acc: 0.9266
    Epoch 95/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2533 - acc: 0.9266
    Epoch 96/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 97/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 98/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 99/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 100/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 101/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 102/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 103/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 104/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 105/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 106/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 107/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 108/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 109/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 110/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 111/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 112/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 113/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 114/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 115/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2523 - acc: 0.9266
    Epoch 116/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 117/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 118/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 119/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 120/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 121/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 122/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 123/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 124/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 125/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 126/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 127/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 128/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 129/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 130/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 131/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 132/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 133/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 134/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 135/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2517 - acc: 0.9266
    Epoch 136/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 137/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 138/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 139/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 140/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 141/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 142/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 143/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 144/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 145/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 146/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 147/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 148/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 149/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 150/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 151/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 152/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 153/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 154/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 155/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 156/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 157/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 158/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 159/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 160/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 161/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 162/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 163/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 164/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 165/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 166/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 167/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 168/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 169/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 170/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 171/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 172/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 173/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2502 - acc: 0.9266
    Epoch 174/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 175/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 176/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 177/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 178/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 179/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 180/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 181/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 182/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 183/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 184/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 185/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 186/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 187/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 188/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 189/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 190/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 191/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 192/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 193/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 194/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 195/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2483 - acc: 0.9266
    Epoch 196/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 197/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 198/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 199/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 200/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    
     AUC: training is 0.7006532852667751; validation is 0.7215206512220411
    
     model 3.4 starts to fit ... 
    
    Epoch 1/200
    45000/45000 [==============================] - 1s 18us/step - loss: 0.6978 - acc: 0.9044
    Epoch 2/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6879 - acc: 0.9266
    Epoch 3/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6756 - acc: 0.9266
    Epoch 4/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6246 - acc: 0.9266
    Epoch 5/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.4322 - acc: 0.9266
    Epoch 6/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3391 - acc: 0.9266
    Epoch 7/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3211 - acc: 0.9266
    Epoch 8/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3140 - acc: 0.9266
    Epoch 9/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3065 - acc: 0.9266
    Epoch 10/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3000 - acc: 0.9266
    Epoch 11/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2947 - acc: 0.9266
    Epoch 12/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2912 - acc: 0.9266
    Epoch 13/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2882 - acc: 0.9266
    Epoch 14/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2857 - acc: 0.9266
    Epoch 15/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2834 - acc: 0.9266
    Epoch 16/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2804 - acc: 0.9266
    Epoch 17/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2794 - acc: 0.9266
    Epoch 18/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2790 - acc: 0.9266
    Epoch 19/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2762 - acc: 0.9266
    Epoch 20/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2745 - acc: 0.9266
    Epoch 21/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2746 - acc: 0.9266
    Epoch 22/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2734 - acc: 0.9266
    Epoch 23/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2730 - acc: 0.9266
    Epoch 24/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2717 - acc: 0.9266
    Epoch 25/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2699 - acc: 0.9266
    Epoch 26/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2691 - acc: 0.9266
    Epoch 27/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2691 - acc: 0.9266
    Epoch 28/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2687 - acc: 0.9266
    Epoch 29/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2660 - acc: 0.9266
    Epoch 30/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2676 - acc: 0.9266
    Epoch 31/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2665 - acc: 0.9266
    Epoch 32/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2651 - acc: 0.9266
    Epoch 33/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2641 - acc: 0.9266
    Epoch 34/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2637 - acc: 0.9266
    Epoch 35/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2638 - acc: 0.9266
    Epoch 36/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2630 - acc: 0.9266
    Epoch 37/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2624 - acc: 0.9266
    Epoch 38/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2610 - acc: 0.9266
    Epoch 39/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2614 - acc: 0.9266
    Epoch 40/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2594 - acc: 0.9266
    Epoch 41/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2599 - acc: 0.9266
    Epoch 42/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2603 - acc: 0.9266
    Epoch 43/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2602 - acc: 0.9266
    Epoch 44/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2588 - acc: 0.9266
    Epoch 45/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2591 - acc: 0.9266
    Epoch 46/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2589 - acc: 0.9266
    Epoch 47/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2584 - acc: 0.9266
    Epoch 48/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2573 - acc: 0.9266
    Epoch 49/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2577 - acc: 0.9266
    Epoch 50/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2575 - acc: 0.9266
    Epoch 51/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 52/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2565 - acc: 0.9266
    Epoch 53/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2561 - acc: 0.9266
    Epoch 54/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2562 - acc: 0.9266
    Epoch 55/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2558 - acc: 0.9266
    Epoch 56/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 57/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2561 - acc: 0.9266
    Epoch 58/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 59/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2563 - acc: 0.9266
    Epoch 60/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2552 - acc: 0.9266
    Epoch 61/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2548 - acc: 0.9266
    Epoch 62/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2558 - acc: 0.9266
    Epoch 63/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 64/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 65/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 66/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 67/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 68/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 69/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 70/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 71/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 72/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2540 - acc: 0.9266
    Epoch 73/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 74/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 75/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 76/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 77/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 78/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 79/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 80/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 81/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 82/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 83/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 84/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 85/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 86/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 87/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 88/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 89/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 90/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 91/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 92/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 93/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 94/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 95/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 96/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 97/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 98/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 99/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 100/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 101/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 102/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 103/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 104/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 105/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 106/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 107/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 108/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 109/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 110/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 111/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 112/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 113/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 114/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 115/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 116/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 117/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 118/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 119/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 120/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 121/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 122/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 123/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 124/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 125/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 126/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 127/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 128/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 129/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 130/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 131/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 132/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 133/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 134/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 135/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 136/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 137/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 138/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 139/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 140/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 141/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 142/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 143/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 144/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 145/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 146/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 147/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 148/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 149/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 150/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 151/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 152/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 153/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 154/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 155/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 156/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 157/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 158/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 159/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 160/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 161/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 162/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 163/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 164/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 165/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 166/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 167/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 168/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 169/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 170/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 171/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 172/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 173/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 174/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 175/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 176/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 177/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 178/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 179/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 180/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 181/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 182/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 183/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 184/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 185/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 186/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 187/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 188/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 189/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 190/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 191/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 192/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 193/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 194/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 195/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 196/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 197/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 198/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 199/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 200/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    
     AUC: training is 0.6976346456572919; validation is 0.7210416212092965
    
     model 4.4 starts to fit ... 
    
    Epoch 1/200
    45000/45000 [==============================] - 1s 19us/step - loss: 0.6979 - acc: 0.9138
    Epoch 2/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6881 - acc: 0.9266
    Epoch 3/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6770 - acc: 0.9266
    Epoch 4/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6367 - acc: 0.9266
    Epoch 5/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.4603 - acc: 0.9266
    Epoch 6/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3357 - acc: 0.9266
    Epoch 7/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3239 - acc: 0.9266
    Epoch 8/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3143 - acc: 0.9266
    Epoch 9/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3056 - acc: 0.9266
    Epoch 10/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2994 - acc: 0.9266
    Epoch 11/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2948 - acc: 0.9266
    Epoch 12/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2901 - acc: 0.9266
    Epoch 13/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2869 - acc: 0.9266
    Epoch 14/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2842 - acc: 0.9266
    Epoch 15/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2813 - acc: 0.9266
    Epoch 16/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2807 - acc: 0.9266
    Epoch 17/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2792 - acc: 0.9266
    Epoch 18/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2779 - acc: 0.9266
    Epoch 19/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2761 - acc: 0.9266
    Epoch 20/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2760 - acc: 0.9266
    Epoch 21/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2735 - acc: 0.9266
    Epoch 22/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2737 - acc: 0.9266
    Epoch 23/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2717 - acc: 0.9266
    Epoch 24/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2712 - acc: 0.9266
    Epoch 25/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2700 - acc: 0.9266
    Epoch 26/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2704 - acc: 0.9266
    Epoch 27/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2694 - acc: 0.9266
    Epoch 28/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 29/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 30/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2672 - acc: 0.9266
    Epoch 31/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2663 - acc: 0.9266
    Epoch 32/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2661 - acc: 0.9266
    Epoch 33/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2648 - acc: 0.9266
    Epoch 34/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2646 - acc: 0.9266
    Epoch 35/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2642 - acc: 0.9266
    Epoch 36/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2631 - acc: 0.9266
    Epoch 37/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2632 - acc: 0.9266
    Epoch 38/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2624 - acc: 0.9266
    Epoch 39/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2628 - acc: 0.9266
    Epoch 40/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2622 - acc: 0.9266
    Epoch 41/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2623 - acc: 0.9266
    Epoch 42/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2601 - acc: 0.9266
    Epoch 43/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2600 - acc: 0.9266
    Epoch 44/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2596 - acc: 0.9266
    Epoch 45/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2595 - acc: 0.9266
    Epoch 46/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2603 - acc: 0.9266
    Epoch 47/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2592 - acc: 0.9266
    Epoch 48/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2594 - acc: 0.9266
    Epoch 49/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2592 - acc: 0.9266
    Epoch 50/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2589 - acc: 0.9266
    Epoch 51/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2576 - acc: 0.9266
    Epoch 52/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2579 - acc: 0.9266
    Epoch 53/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2582 - acc: 0.9266
    Epoch 54/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2571 - acc: 0.9266
    Epoch 55/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2580 - acc: 0.9266
    Epoch 56/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2573 - acc: 0.9266
    Epoch 57/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2573 - acc: 0.9266
    Epoch 58/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 59/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2571 - acc: 0.9266
    Epoch 60/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 61/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2564 - acc: 0.9266
    Epoch 62/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2558 - acc: 0.9266
    Epoch 63/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2559 - acc: 0.9266
    Epoch 64/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2558 - acc: 0.9266
    Epoch 65/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2552 - acc: 0.9266
    Epoch 66/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2553 - acc: 0.9266
    Epoch 67/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2552 - acc: 0.9266
    Epoch 68/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 69/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 70/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 71/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 72/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 73/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2544 - acc: 0.9266
    Epoch 74/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2554 - acc: 0.9266
    Epoch 75/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 76/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 77/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 78/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 79/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 80/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 81/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 82/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 83/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 84/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 85/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 86/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 87/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 88/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 89/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 90/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 91/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 92/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 93/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 94/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 95/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 96/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 97/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 98/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 99/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 100/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 101/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 102/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 103/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 104/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 105/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 106/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 107/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 108/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 109/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 110/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 111/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 112/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 113/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 114/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 115/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 116/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 117/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 118/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 119/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 120/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 121/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 122/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 123/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 124/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 125/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 126/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 127/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 128/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 129/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 130/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 131/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 132/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 133/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 134/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 135/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 136/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 137/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 138/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 139/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 140/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 141/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 142/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 143/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 144/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 145/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 146/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 147/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 148/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 149/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 150/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 151/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 152/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 153/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 154/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 155/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 156/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 157/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 158/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 159/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 160/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 161/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 162/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 163/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 164/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 165/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 166/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 167/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 168/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 169/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 170/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 171/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 172/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 173/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 174/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 175/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 176/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 177/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 178/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 179/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 180/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 181/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 182/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 183/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 184/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 185/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 186/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 187/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 188/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 189/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 190/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 191/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 192/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 193/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 194/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 195/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 196/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 197/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 198/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 199/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 200/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    
     AUC: training is 0.70300695980591; validation is 0.6902631342148584
    
     model 5.4 starts to fit ... 
    
    Epoch 1/200
    45000/45000 [==============================] - 1s 19us/step - loss: 0.6978 - acc: 0.9109
    Epoch 2/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6879 - acc: 0.9266
    Epoch 3/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6740 - acc: 0.9266
    Epoch 4/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6099 - acc: 0.9266
    Epoch 5/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.4109 - acc: 0.9266
    Epoch 6/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3380 - acc: 0.9266
    Epoch 7/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3224 - acc: 0.9266
    Epoch 8/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3158 - acc: 0.9266
    Epoch 9/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3076 - acc: 0.9266
    Epoch 10/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3028 - acc: 0.9266
    Epoch 11/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2964 - acc: 0.9266
    Epoch 12/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2937 - acc: 0.9266
    Epoch 13/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2901 - acc: 0.9266
    Epoch 14/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2873 - acc: 0.9266
    Epoch 15/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2852 - acc: 0.9266
    Epoch 16/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2821 - acc: 0.9266
    Epoch 17/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2819 - acc: 0.9266
    Epoch 18/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2797 - acc: 0.9266
    Epoch 19/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2778 - acc: 0.9266
    Epoch 20/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2769 - acc: 0.9266
    Epoch 21/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2759 - acc: 0.9266
    Epoch 22/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2752 - acc: 0.9266
    Epoch 23/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2747 - acc: 0.9266
    Epoch 24/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2728 - acc: 0.9266
    Epoch 25/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2717 - acc: 0.9266
    Epoch 26/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2724 - acc: 0.9266
    Epoch 27/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2704 - acc: 0.9266
    Epoch 28/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2700 - acc: 0.9266
    Epoch 29/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2706 - acc: 0.9266
    Epoch 30/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2702 - acc: 0.9266
    Epoch 31/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 32/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2684 - acc: 0.9266
    Epoch 33/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2674 - acc: 0.9266
    Epoch 34/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 35/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2660 - acc: 0.9266
    Epoch 36/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2663 - acc: 0.9266
    Epoch 37/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2660 - acc: 0.9266
    Epoch 38/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2650 - acc: 0.9266
    Epoch 39/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2647 - acc: 0.9266
    Epoch 40/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2648 - acc: 0.9266
    Epoch 41/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2641 - acc: 0.9266
    Epoch 42/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2639 - acc: 0.9266
    Epoch 43/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2627 - acc: 0.9266
    Epoch 44/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2621 - acc: 0.9266
    Epoch 45/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2616 - acc: 0.9266
    Epoch 46/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2620 - acc: 0.9266
    Epoch 47/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2599 - acc: 0.9266
    Epoch 48/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2602 - acc: 0.9266
    Epoch 49/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2605 - acc: 0.9266
    Epoch 50/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2598 - acc: 0.9266
    Epoch 51/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2588 - acc: 0.9266
    Epoch 52/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2589 - acc: 0.9266
    Epoch 53/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2587 - acc: 0.9266
    Epoch 54/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2582 - acc: 0.9266
    Epoch 55/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2582 - acc: 0.9266
    Epoch 56/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2582 - acc: 0.9266
    Epoch 57/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2575 - acc: 0.9266
    Epoch 58/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 59/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2575 - acc: 0.9266
    Epoch 60/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2574 - acc: 0.9266
    Epoch 61/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2566 - acc: 0.9266
    Epoch 62/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2566 - acc: 0.9266
    Epoch 63/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2561 - acc: 0.9266
    Epoch 64/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2560 - acc: 0.9266
    Epoch 65/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 66/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2558 - acc: 0.9266
    Epoch 67/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 68/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 69/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 70/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2556 - acc: 0.9266
    Epoch 71/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 72/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 73/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 74/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2540 - acc: 0.9266
    Epoch 75/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 76/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 77/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 78/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 79/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 80/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 81/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 82/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 83/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 84/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 85/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2533 - acc: 0.9266
    Epoch 86/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 87/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 88/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 89/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 90/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 91/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 92/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 93/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 94/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2535 - acc: 0.9266
    Epoch 95/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 96/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 97/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 98/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 99/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 100/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 101/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 102/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 103/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 104/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 105/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 106/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 107/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 108/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2521 - acc: 0.9266
    Epoch 109/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 110/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 111/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 112/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 113/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 114/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 115/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 116/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 117/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 118/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 119/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 120/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 121/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 122/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 123/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 124/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 125/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 126/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 127/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 128/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 129/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2515 - acc: 0.9266
    Epoch 130/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 131/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 132/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 133/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 134/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 135/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 136/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 137/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 138/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 139/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 140/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2517 - acc: 0.9266
    Epoch 141/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 142/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 143/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 144/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 145/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 146/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 147/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 148/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 149/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 150/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 151/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 152/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 153/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 154/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 155/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2504 - acc: 0.9266
    Epoch 156/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 157/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 158/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 159/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 160/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 161/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 162/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 163/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 164/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 165/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 166/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 167/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 168/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 169/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 170/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 171/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 172/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 173/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 174/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 175/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 176/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 177/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 178/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 179/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 180/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 181/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 182/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 183/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 184/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 185/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 186/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 187/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 188/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 189/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 190/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 191/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 192/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 193/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 194/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2490 - acc: 0.9266
    Epoch 195/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2492 - acc: 0.9266
    Epoch 196/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2495 - acc: 0.9266
    Epoch 197/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 198/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2495 - acc: 0.9266
    Epoch 199/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 200/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    
     AUC: training is 0.6995865842996808; validation is 0.7221208355412627
    
     model 6.4 starts to fit ... 
    
    Epoch 1/200
    45000/45000 [==============================] - 1s 20us/step - loss: 0.6980 - acc: 0.8881
    Epoch 2/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6881 - acc: 0.9266
    Epoch 3/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6756 - acc: 0.9266
    Epoch 4/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6261 - acc: 0.9266
    Epoch 5/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.4458 - acc: 0.9266
    Epoch 6/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3360 - acc: 0.9266
    Epoch 7/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3253 - acc: 0.9266
    Epoch 8/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3144 - acc: 0.9266
    Epoch 9/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3066 - acc: 0.9266
    Epoch 10/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2997 - acc: 0.9266
    Epoch 11/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2949 - acc: 0.9266
    Epoch 12/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2905 - acc: 0.9266
    Epoch 13/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2880 - acc: 0.9266
    Epoch 14/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2856 - acc: 0.9266
    Epoch 15/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2821 - acc: 0.9266
    Epoch 16/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2816 - acc: 0.9266
    Epoch 17/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2797 - acc: 0.9266
    Epoch 18/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2770 - acc: 0.9266
    Epoch 19/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2769 - acc: 0.9266
    Epoch 20/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2765 - acc: 0.9266
    Epoch 21/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2759 - acc: 0.9266
    Epoch 22/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2742 - acc: 0.9266
    Epoch 23/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2728 - acc: 0.9266
    Epoch 24/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2715 - acc: 0.9266
    Epoch 25/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2722 - acc: 0.9266
    Epoch 26/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2710 - acc: 0.9266
    Epoch 27/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2694 - acc: 0.9266
    Epoch 28/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2696 - acc: 0.9266
    Epoch 29/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2690 - acc: 0.9266
    Epoch 30/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2681 - acc: 0.9266
    Epoch 31/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2673 - acc: 0.9266
    Epoch 32/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2670 - acc: 0.9266
    Epoch 33/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2657 - acc: 0.9266
    Epoch 34/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2667 - acc: 0.9266
    Epoch 35/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2649 - acc: 0.9266
    Epoch 36/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2646 - acc: 0.9266
    Epoch 37/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2644 - acc: 0.9266
    Epoch 38/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2649 - acc: 0.9266
    Epoch 39/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2634 - acc: 0.9266
    Epoch 40/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2636 - acc: 0.9266
    Epoch 41/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2616 - acc: 0.9266
    Epoch 42/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2619 - acc: 0.9266
    Epoch 43/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2615 - acc: 0.9266
    Epoch 44/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2608 - acc: 0.9266
    Epoch 45/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2601 - acc: 0.9266
    Epoch 46/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2603 - acc: 0.9266
    Epoch 47/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2606 - acc: 0.9266
    Epoch 48/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2594 - acc: 0.9266
    Epoch 49/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2594 - acc: 0.9266
    Epoch 50/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2607 - acc: 0.9266
    Epoch 51/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2583 - acc: 0.9266
    Epoch 52/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2593 - acc: 0.9266
    Epoch 53/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2577 - acc: 0.9266
    Epoch 54/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2585 - acc: 0.9266
    Epoch 55/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2578 - acc: 0.9266
    Epoch 56/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2573 - acc: 0.9266
    Epoch 57/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2574 - acc: 0.9266
    Epoch 58/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2578 - acc: 0.9266
    Epoch 59/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2570 - acc: 0.9266
    Epoch 60/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2568 - acc: 0.9266
    Epoch 61/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 62/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2565 - acc: 0.9266
    Epoch 63/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2564 - acc: 0.9266
    Epoch 64/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2566 - acc: 0.9266
    Epoch 65/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 66/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 67/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 68/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 69/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2552 - acc: 0.9266
    Epoch 70/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2555 - acc: 0.9266
    Epoch 71/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 72/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2551 - acc: 0.9266
    Epoch 73/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2545 - acc: 0.9266
    Epoch 74/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2548 - acc: 0.9266
    Epoch 75/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2545 - acc: 0.9266
    Epoch 76/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 77/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 78/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 79/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 80/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 81/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 82/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2547 - acc: 0.9266
    Epoch 83/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 84/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 85/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 86/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 87/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 88/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 89/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 90/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 91/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2528 - acc: 0.9266
    Epoch 92/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 93/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 94/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2533 - acc: 0.9266
    Epoch 95/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2531 - acc: 0.9266
    Epoch 96/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 97/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 98/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2533 - acc: 0.9266
    Epoch 99/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 100/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 101/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 102/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2524 - acc: 0.9266
    Epoch 103/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 104/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 105/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 106/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 107/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 108/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 109/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 110/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2527 - acc: 0.9266
    Epoch 111/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 112/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 113/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 114/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 115/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 116/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 117/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 118/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 119/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 120/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 121/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 122/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 123/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 124/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 125/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 126/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 127/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 128/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 129/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 130/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 131/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 132/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 133/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 134/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 135/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 136/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 137/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2517 - acc: 0.9266
    Epoch 138/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 139/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 140/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 141/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 142/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 143/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 144/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 145/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 146/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 147/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 148/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 149/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 150/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 151/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 152/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 153/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2512 - acc: 0.9266
    Epoch 154/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 155/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 156/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 157/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 158/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 159/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2516 - acc: 0.9266
    Epoch 160/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 161/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2507 - acc: 0.9266
    Epoch 162/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 163/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 164/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 165/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 166/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 167/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2517 - acc: 0.9266
    Epoch 168/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2511 - acc: 0.9266
    Epoch 169/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 170/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 171/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 172/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 173/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 174/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 175/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 176/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 177/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2511 - acc: 0.9266
    Epoch 178/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 179/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 180/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 181/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2511 - acc: 0.9266
    Epoch 182/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 183/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 184/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 185/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 186/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 187/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 188/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 189/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 190/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 191/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2516 - acc: 0.9266
    Epoch 192/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 193/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 194/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 195/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 196/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 197/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 198/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 199/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 200/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    
     AUC: training is 0.6985295965460583; validation is 0.6889127930125724
    
     model 7.4 starts to fit ... 
    
    Epoch 1/200
    45000/45000 [==============================] - 1s 21us/step - loss: 0.6979 - acc: 0.8782
    Epoch 2/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6882 - acc: 0.9266
    Epoch 3/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6778 - acc: 0.9266
    Epoch 4/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.6454 - acc: 0.9266
    Epoch 5/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.5004 - acc: 0.9266
    Epoch 6/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3420 - acc: 0.9266
    Epoch 7/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3318 - acc: 0.9266
    Epoch 8/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3189 - acc: 0.9266
    Epoch 9/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3114 - acc: 0.9266
    Epoch 10/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.3059 - acc: 0.9266
    Epoch 11/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2999 - acc: 0.9266
    Epoch 12/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2955 - acc: 0.9266
    Epoch 13/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2923 - acc: 0.9266
    Epoch 14/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2876 - acc: 0.9266
    Epoch 15/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2847 - acc: 0.9266
    Epoch 16/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2822 - acc: 0.9266
    Epoch 17/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2807 - acc: 0.9266
    Epoch 18/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2788 - acc: 0.9266
    Epoch 19/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2774 - acc: 0.9266
    Epoch 20/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2757 - acc: 0.9266
    Epoch 21/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2747 - acc: 0.9266
    Epoch 22/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2740 - acc: 0.9266
    Epoch 23/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2728 - acc: 0.9266
    Epoch 24/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2711 - acc: 0.9266
    Epoch 25/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2709 - acc: 0.9266
    Epoch 26/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2700 - acc: 0.9266
    Epoch 27/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2674 - acc: 0.9266
    Epoch 28/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2688 - acc: 0.9266
    Epoch 29/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 30/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2666 - acc: 0.9266
    Epoch 31/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2668 - acc: 0.9266
    Epoch 32/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2653 - acc: 0.9266
    Epoch 33/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2643 - acc: 0.9266
    Epoch 34/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2647 - acc: 0.9266
    Epoch 35/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2639 - acc: 0.9266
    Epoch 36/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2638 - acc: 0.9266
    Epoch 37/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2622 - acc: 0.9266
    Epoch 38/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2622 - acc: 0.9266
    Epoch 39/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2613 - acc: 0.9266
    Epoch 40/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2610 - acc: 0.9266
    Epoch 41/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2611 - acc: 0.9266
    Epoch 42/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2605 - acc: 0.9266
    Epoch 43/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2601 - acc: 0.9266
    Epoch 44/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2597 - acc: 0.9266
    Epoch 45/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2588 - acc: 0.9266
    Epoch 46/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2585 - acc: 0.9266
    Epoch 47/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2587 - acc: 0.9266
    Epoch 48/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2573 - acc: 0.9266
    Epoch 49/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2565 - acc: 0.9266
    Epoch 50/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 51/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2570 - acc: 0.9266
    Epoch 52/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2566 - acc: 0.9266
    Epoch 53/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2563 - acc: 0.9266
    Epoch 54/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2569 - acc: 0.9266
    Epoch 55/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2569 - acc: 0.9266
    Epoch 56/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2552 - acc: 0.9266
    Epoch 57/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2566 - acc: 0.9266
    Epoch 58/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2557 - acc: 0.9266
    Epoch 59/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 60/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2556 - acc: 0.9266
    Epoch 61/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 62/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2552 - acc: 0.9266
    Epoch 63/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2546 - acc: 0.9266
    Epoch 64/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 65/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2552 - acc: 0.9266
    Epoch 66/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 67/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 68/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 69/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2542 - acc: 0.9266
    Epoch 70/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2539 - acc: 0.9266
    Epoch 71/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 72/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2537 - acc: 0.9266
    Epoch 73/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 74/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2533 - acc: 0.9266
    Epoch 75/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 76/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2538 - acc: 0.9266
    Epoch 77/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 78/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 79/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 80/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2523 - acc: 0.9266
    Epoch 81/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 82/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 83/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2531 - acc: 0.9266
    Epoch 84/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2528 - acc: 0.9266
    Epoch 85/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 86/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 87/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 88/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 89/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 90/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 91/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 92/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 93/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2512 - acc: 0.9266
    Epoch 94/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 95/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 96/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 97/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 98/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2514 - acc: 0.9266
    Epoch 99/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 100/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 101/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 102/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 103/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 104/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 105/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2514 - acc: 0.9266
    Epoch 106/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 107/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 108/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 109/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 110/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 111/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 112/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 113/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 114/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 115/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 116/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 117/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 118/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 119/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 120/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 121/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 122/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 123/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 124/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 125/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 126/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 127/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2502 - acc: 0.9266
    Epoch 128/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 129/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 130/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2499 - acc: 0.9266
    Epoch 131/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 132/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2500 - acc: 0.9266
    Epoch 133/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2501 - acc: 0.9266
    Epoch 134/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 135/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2497 - acc: 0.9266
    Epoch 136/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 137/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 138/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 139/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2493 - acc: 0.9266
    Epoch 140/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 141/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 142/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 143/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 144/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 145/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 146/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 147/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 148/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 149/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 150/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 151/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 152/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 153/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 154/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 155/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 156/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 157/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 158/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 159/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 160/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 161/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 162/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 163/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 164/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 165/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2485 - acc: 0.9266
    Epoch 166/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 167/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2474 - acc: 0.9266
    Epoch 168/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 169/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2476 - acc: 0.9266
    Epoch 170/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 171/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2483 - acc: 0.9266
    Epoch 172/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 173/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 174/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 175/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2475 - acc: 0.9266
    Epoch 176/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 177/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 178/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2476 - acc: 0.9266
    Epoch 179/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 180/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2481 - acc: 0.9266
    Epoch 181/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2478 - acc: 0.9266
    Epoch 182/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 183/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2472 - acc: 0.9266
    Epoch 184/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2477 - acc: 0.9266
    Epoch 185/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2478 - acc: 0.9266
    Epoch 186/200
    45000/45000 [==============================] - 0s 5us/step - loss: 0.2475 - acc: 0.9266
    Epoch 187/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 188/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2476 - acc: 0.9266
    Epoch 189/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 190/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2483 - acc: 0.9266
    Epoch 191/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 192/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 193/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 194/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2476 - acc: 0.9266
    Epoch 195/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2475 - acc: 0.9266
    Epoch 196/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2476 - acc: 0.9266
    Epoch 197/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 198/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2473 - acc: 0.9266
    Epoch 199/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2472 - acc: 0.9266
    Epoch 200/200
    45000/45000 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    
     AUC: training is 0.7067885311513135; validation is 0.67219614529342
    
     model 8.4 starts to fit ... 
    
    Epoch 1/200
    45001/45001 [==============================] - 1s 22us/step - loss: 0.6978 - acc: 0.9058
    Epoch 2/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.6879 - acc: 0.9266
    Epoch 3/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.6756 - acc: 0.9266
    Epoch 4/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.6281 - acc: 0.9266
    Epoch 5/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.4487 - acc: 0.9266
    Epoch 6/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.3187 - acc: 0.9266
    Epoch 7/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.3169 - acc: 0.9266
    Epoch 8/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.3056 - acc: 0.9266
    Epoch 9/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.3008 - acc: 0.9266
    Epoch 10/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2971 - acc: 0.9266
    Epoch 11/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2938 - acc: 0.9266
    Epoch 12/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2919 - acc: 0.9266
    Epoch 13/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2889 - acc: 0.9266
    Epoch 14/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2860 - acc: 0.9266
    Epoch 15/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2844 - acc: 0.9266
    Epoch 16/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2831 - acc: 0.9266
    Epoch 17/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2810 - acc: 0.9266
    Epoch 18/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2778 - acc: 0.9266
    Epoch 19/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2776 - acc: 0.9266
    Epoch 20/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2747 - acc: 0.9266
    Epoch 21/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2740 - acc: 0.9266
    Epoch 22/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2731 - acc: 0.9266
    Epoch 23/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2718 - acc: 0.9266
    Epoch 24/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2697 - acc: 0.9266
    Epoch 25/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2693 - acc: 0.9266
    Epoch 26/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2693 - acc: 0.9266
    Epoch 27/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2668 - acc: 0.9266
    Epoch 28/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2681 - acc: 0.9266
    Epoch 29/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2664 - acc: 0.9266
    Epoch 30/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2653 - acc: 0.9266
    Epoch 31/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2647 - acc: 0.9266
    Epoch 32/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2633 - acc: 0.9266
    Epoch 33/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2641 - acc: 0.9266
    Epoch 34/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2633 - acc: 0.9266
    Epoch 35/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2622 - acc: 0.9266
    Epoch 36/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2619 - acc: 0.9266
    Epoch 37/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2609 - acc: 0.9266
    Epoch 38/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2606 - acc: 0.9266
    Epoch 39/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2612 - acc: 0.9266
    Epoch 40/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2600 - acc: 0.9266
    Epoch 41/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2598 - acc: 0.9266
    Epoch 42/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2593 - acc: 0.9266
    Epoch 43/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2590 - acc: 0.9266
    Epoch 44/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2588 - acc: 0.9266
    Epoch 45/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2586 - acc: 0.9266
    Epoch 46/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2583 - acc: 0.9266
    Epoch 47/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2581 - acc: 0.9266
    Epoch 48/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2567 - acc: 0.9266
    Epoch 49/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2580 - acc: 0.9266
    Epoch 50/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2567 - acc: 0.9266
    Epoch 51/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2573 - acc: 0.9266
    Epoch 52/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2568 - acc: 0.9266
    Epoch 53/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2564 - acc: 0.9266
    Epoch 54/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2558 - acc: 0.9266
    Epoch 55/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 56/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2561 - acc: 0.9266
    Epoch 57/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2552 - acc: 0.9266
    Epoch 58/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2548 - acc: 0.9266
    Epoch 59/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2555 - acc: 0.9266
    Epoch 60/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2553 - acc: 0.9266
    Epoch 61/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2554 - acc: 0.9266
    Epoch 62/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2548 - acc: 0.9266
    Epoch 63/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 64/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 65/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2542 - acc: 0.9266
    Epoch 66/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2536 - acc: 0.9266
    Epoch 67/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2544 - acc: 0.9266
    Epoch 68/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2549 - acc: 0.9266
    Epoch 69/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2539 - acc: 0.9266
    Epoch 70/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2541 - acc: 0.9266
    Epoch 71/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 72/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 73/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2534 - acc: 0.9266
    Epoch 74/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2532 - acc: 0.9266
    Epoch 75/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 76/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 77/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 78/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 79/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 80/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 81/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 82/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 83/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2519 - acc: 0.9266
    Epoch 84/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2523 - acc: 0.9266
    Epoch 85/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 86/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 87/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 88/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2521 - acc: 0.9266
    Epoch 89/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 90/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2519 - acc: 0.9266
    Epoch 91/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2516 - acc: 0.9266
    Epoch 92/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 93/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2523 - acc: 0.9266
    Epoch 94/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2513 - acc: 0.9266
    Epoch 95/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 96/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 97/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2514 - acc: 0.9266
    Epoch 98/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2521 - acc: 0.9266
    Epoch 99/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 100/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 101/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 102/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 103/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 104/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 105/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 106/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 107/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 108/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 109/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 110/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 111/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 112/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 113/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 114/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 115/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 116/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 117/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 118/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 119/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 120/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 121/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 122/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 123/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 124/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2499 - acc: 0.9266
    Epoch 125/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 126/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 127/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2501 - acc: 0.9266
    Epoch 128/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 129/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 130/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 131/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 132/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 133/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 134/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 135/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 136/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 137/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 138/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 139/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 140/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 141/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 142/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 143/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 144/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 145/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2498 - acc: 0.9266
    Epoch 146/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 147/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2488 - acc: 0.9266
    Epoch 148/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 149/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2495 - acc: 0.9266
    Epoch 150/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 151/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 152/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 153/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2489 - acc: 0.9266
    Epoch 154/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2494 - acc: 0.9266
    Epoch 155/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 156/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 157/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2491 - acc: 0.9266
    Epoch 158/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 159/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2486 - acc: 0.9266
    Epoch 160/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2486 - acc: 0.9266
    Epoch 161/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 162/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 163/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 164/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 165/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 166/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 167/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 168/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2484 - acc: 0.9266
    Epoch 169/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 170/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2485 - acc: 0.9266
    Epoch 171/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 172/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 173/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2486 - acc: 0.9266
    Epoch 174/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2487 - acc: 0.9266
    Epoch 175/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2488 - acc: 0.9266
    Epoch 176/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2485 - acc: 0.9266
    Epoch 177/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 178/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 179/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 180/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 181/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 182/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 183/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 184/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 185/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2479 - acc: 0.9266
    Epoch 186/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2484 - acc: 0.9266
    Epoch 187/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 188/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2482 - acc: 0.9266
    Epoch 189/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 190/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 191/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 192/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2485 - acc: 0.9266
    Epoch 193/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2484 - acc: 0.9266
    Epoch 194/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2481 - acc: 0.9266
    Epoch 195/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2484 - acc: 0.9266
    Epoch 196/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 197/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2483 - acc: 0.9266
    Epoch 198/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2484 - acc: 0.9266
    Epoch 199/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2481 - acc: 0.9266
    Epoch 200/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2482 - acc: 0.9266
    
     AUC: training is 0.7064528296738702; validation is 0.658901704997341
    
     model 9.4 starts to fit ... 
    
    Epoch 1/200
    45001/45001 [==============================] - 1s 23us/step - loss: 0.6980 - acc: 0.8881
    Epoch 2/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.6882 - acc: 0.9266
    Epoch 3/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.6771 - acc: 0.9266
    Epoch 4/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.6347 - acc: 0.9266
    Epoch 5/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.4534 - acc: 0.9266
    Epoch 6/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.3203 - acc: 0.9266
    Epoch 7/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.3102 - acc: 0.9266
    Epoch 8/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.3024 - acc: 0.9266
    Epoch 9/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2973 - acc: 0.9266
    Epoch 10/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2943 - acc: 0.9266
    Epoch 11/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2900 - acc: 0.9266
    Epoch 12/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2883 - acc: 0.9266
    Epoch 13/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2843 - acc: 0.9266
    Epoch 14/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2846 - acc: 0.9266
    Epoch 15/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2827 - acc: 0.9266
    Epoch 16/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2799 - acc: 0.9266
    Epoch 17/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2784 - acc: 0.9266
    Epoch 18/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2769 - acc: 0.9266
    Epoch 19/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2735 - acc: 0.9266
    Epoch 20/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2740 - acc: 0.9266
    Epoch 21/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2724 - acc: 0.9266
    Epoch 22/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2710 - acc: 0.9266
    Epoch 23/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2701 - acc: 0.9266
    Epoch 24/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2690 - acc: 0.9266
    Epoch 25/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2685 - acc: 0.9266
    Epoch 26/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2658 - acc: 0.9266
    Epoch 27/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2678 - acc: 0.9266
    Epoch 28/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2659 - acc: 0.9266
    Epoch 29/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2651 - acc: 0.9266
    Epoch 30/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2644 - acc: 0.9266
    Epoch 31/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2636 - acc: 0.9266
    Epoch 32/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2622 - acc: 0.9266
    Epoch 33/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2620 - acc: 0.9266
    Epoch 34/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2624 - acc: 0.9266
    Epoch 35/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2624 - acc: 0.9266
    Epoch 36/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2598 - acc: 0.9266
    Epoch 37/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2604 - acc: 0.9266
    Epoch 38/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2600 - acc: 0.9266
    Epoch 39/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2596 - acc: 0.9266
    Epoch 40/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2585 - acc: 0.9266
    Epoch 41/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2593 - acc: 0.9266
    Epoch 42/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2576 - acc: 0.9266
    Epoch 43/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2566 - acc: 0.9266
    Epoch 44/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 45/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2572 - acc: 0.9266
    Epoch 46/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2570 - acc: 0.9266
    Epoch 47/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2566 - acc: 0.9266
    Epoch 48/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2558 - acc: 0.9266
    Epoch 49/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2558 - acc: 0.9266
    Epoch 50/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2562 - acc: 0.9266
    Epoch 51/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2563 - acc: 0.9266
    Epoch 52/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2549 - acc: 0.9266
    Epoch 53/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2550 - acc: 0.9266
    Epoch 54/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2551 - acc: 0.9266
    Epoch 55/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2551 - acc: 0.9266
    Epoch 56/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2548 - acc: 0.9266
    Epoch 57/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2543 - acc: 0.9266
    Epoch 58/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2540 - acc: 0.9266
    Epoch 59/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2548 - acc: 0.9266
    Epoch 60/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2545 - acc: 0.9266
    Epoch 61/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2544 - acc: 0.9266
    Epoch 62/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 63/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 64/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 65/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2540 - acc: 0.9266
    Epoch 66/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2535 - acc: 0.9266
    Epoch 67/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 68/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2536 - acc: 0.9266
    Epoch 69/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2526 - acc: 0.9266
    Epoch 70/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 71/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 72/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 73/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2529 - acc: 0.9266
    Epoch 74/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2526 - acc: 0.9266
    Epoch 75/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2530 - acc: 0.9266
    Epoch 76/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2527 - acc: 0.9266
    Epoch 77/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2524 - acc: 0.9266
    Epoch 78/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 79/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2525 - acc: 0.9266
    Epoch 80/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 81/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 82/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2522 - acc: 0.9266
    Epoch 83/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 84/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2520 - acc: 0.9266
    Epoch 85/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2517 - acc: 0.9266
    Epoch 86/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2515 - acc: 0.9266
    Epoch 87/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2518 - acc: 0.9266
    Epoch 88/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 89/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 90/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 91/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 92/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 93/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 94/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 95/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2511 - acc: 0.9266
    Epoch 96/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2519 - acc: 0.9266
    Epoch 97/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 98/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2508 - acc: 0.9266
    Epoch 99/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 100/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 101/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 102/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2512 - acc: 0.9266
    Epoch 103/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 104/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2509 - acc: 0.9266
    Epoch 105/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 106/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 107/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 108/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2510 - acc: 0.9266
    Epoch 109/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2513 - acc: 0.9266
    Epoch 110/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2516 - acc: 0.9266
    Epoch 111/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 112/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 113/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 114/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2507 - acc: 0.9266
    Epoch 115/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2505 - acc: 0.9266
    Epoch 116/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2498 - acc: 0.9266
    Epoch 117/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 118/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2503 - acc: 0.9266
    Epoch 119/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2502 - acc: 0.9266
    Epoch 120/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 121/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2504 - acc: 0.9266
    Epoch 122/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2502 - acc: 0.9266
    Epoch 123/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 124/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 125/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 126/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2506 - acc: 0.9266
    Epoch 127/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 128/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 129/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 130/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 131/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2505 - acc: 0.9266
    Epoch 132/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 133/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2497 - acc: 0.9266
    Epoch 134/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 135/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2496 - acc: 0.9266
    Epoch 136/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 137/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2495 - acc: 0.9266
    Epoch 138/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 139/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2500 - acc: 0.9266
    Epoch 140/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2489 - acc: 0.9266
    Epoch 141/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2498 - acc: 0.9266
    Epoch 142/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 143/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 144/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 145/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 146/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 147/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 148/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2499 - acc: 0.9266
    Epoch 149/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2498 - acc: 0.9266
    Epoch 150/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 151/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 152/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 153/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 154/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2494 - acc: 0.9266
    Epoch 155/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2493 - acc: 0.9266
    Epoch 156/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2495 - acc: 0.9266
    Epoch 157/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 158/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2492 - acc: 0.9266
    Epoch 159/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 160/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 161/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 162/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2483 - acc: 0.9266
    Epoch 163/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 164/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2479 - acc: 0.9266
    Epoch 165/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2484 - acc: 0.9266
    Epoch 166/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 167/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 168/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2486 - acc: 0.9266
    Epoch 169/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2490 - acc: 0.9266
    Epoch 170/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2485 - acc: 0.9266
    Epoch 171/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2488 - acc: 0.9266
    Epoch 172/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 173/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 174/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 175/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2480 - acc: 0.9266
    Epoch 176/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2488 - acc: 0.9266
    Epoch 177/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 178/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2489 - acc: 0.9266
    Epoch 179/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2487 - acc: 0.9266
    Epoch 180/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2483 - acc: 0.9266
    Epoch 181/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2484 - acc: 0.9266
    Epoch 182/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 183/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2481 - acc: 0.9266
    Epoch 184/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2483 - acc: 0.9266
    Epoch 185/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2479 - acc: 0.9266
    Epoch 186/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 187/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 188/200
    45001/45001 [==============================] - 0s 5us/step - loss: 0.2484 - acc: 0.9266
    Epoch 189/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 190/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2475 - acc: 0.9266
    Epoch 191/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2481 - acc: 0.9266
    Epoch 192/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 193/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 194/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2482 - acc: 0.9266
    Epoch 195/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2481 - acc: 0.9266
    Epoch 196/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2484 - acc: 0.9266
    Epoch 197/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 198/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 199/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2486 - acc: 0.9266
    Epoch 200/200
    45001/45001 [==============================] - 0s 4us/step - loss: 0.2478 - acc: 0.9266
    
     AUC: training is 0.705633024365975; validation is 0.6725856851755115
    training AUC: maximum: 0.7067885311513135; minimun: 0.6976346456572919; mean: 0.7021841964491743; sd: 0.003089806328928281
    validaton AUC: maximum: 0.7221208355412627; minimun: 0.658901704997341; mean: 0.6939695626719757; sd: 0.0213180770593197

