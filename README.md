Trader
===

## Data Preprocess

  I get some price difference amount the `open-high-low-close` of the stock
  ```
  Data['High-Low'] = Data['high'] - Data['low'] #today's High - Low
  Data['High-Close'] = Data['high'] - Data['close'] #today's High - Close
  Data['High-Open'] = Data['high'] - Data['open'] #today's High - Open
  Data['Close-Open'] = Data['close'] - Data['open'] #today's Close - Open
  Data['Close-High'] = Data['close'] - Data['high'] #today's Close - High
  Data['Close-Low'] = Data['close'] - Data['low'] #today's Close - Low
  Data['Open-Close'] = Data['open'] - Data['close'] #todat's Open - Close
  Data['Open-High'] = Data['open'] - Data['high'] #today's Open - High
  ```
  And then I predict if the price of the day after tomorrow is high than tomorrow,
  if higher than I label `1`, if not I label `0`
  ```
  value = pd.Series(Data['open'].shift(-2) - Data['open'].shift(-1), index = Data.index)
  ...
  value[value >= 0] = 1 #0 means rise
  value[value < 0] = 0 #1 means fall
  ```
  Finally, I delete the original feature
  ```
  del(Data['open'])
  del(Data['close'])
  del(Data['high'])
  del(Data['low'])
  ```

## Train_model

I use `cross_validation` to split my data into 50% for train, 50 for test.
And I use SVM with `rbf` kernel.
```
def Trader(train_x, train_y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size = 0.5, random_state = 0)
    clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    score = clf.score(X_test, y_test)
    # print(score)
    predict = clf.predict(X_train)
    return clf
```

## Predict and Action

The testing data also be preprocessed. And my strategy of action is:
**if I predict `1` then I buy without violating the rule, if I predict `0` then I sell without
violating the rule.**
```
def get_action(array):
    global handle
    for index, x in np.ndenumerate(array):
        if handle == 0:
            if x == 1:
                handle = 1
                return 1
            elif x == 0:
                handle = -1
                return -1
        elif handle == 1:
            if x == 1:
                return 0
            elif x == 0:
                handle = 0
                return -1
        elif handle == -1:
            if x == 1:
                handle = 0
                return 1
            elif x == 0:
                return 0
```
