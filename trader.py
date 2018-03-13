
# You can write code above the if-main block.
import pandas as pd
import numpy as np
import csv
from sklearn import cross_validation
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

def load_data(file_name):
    column_names = ['open','high','low','close']
    df = pd.read_csv(file_name, header = None, names = column_names)
    return df

def process(Data):
    value = pd.Series(Data['open'].shift(-2) - Data['open'].shift(-1), index = Data.index)
    Data['High-Low'] = Data['high'] - Data['low'] #today's High - Low
    Data['High-Close'] = Data['high'] - Data['close'] #today's High - Close
    Data['High-Open'] = Data['high'] - Data['open'] #today's High - Open
    Data['Close-Open'] = Data['close'] - Data['open'] #today's Close - Open
    Data['Close-High'] = Data['close'] - Data['high'] #today's Close - High
    Data['Close-Low'] = Data['close'] - Data['low'] #today's Close - Low
    Data['Open-Close'] = Data['open'] - Data['close'] #todat's Open - Close
    Data['Open-High'] = Data['open'] - Data['high'] #today's Open - High
    value = value.dropna()
    value[value >= 0] = 1 #0 means rise
    value[value < 0] = 0 #1 means fall
    del(Data['open'])
    del(Data['close'])
    del(Data['high'])
    del(Data['low'])
    return Data[:len(value)], value

def process_test(Data):
    Data['High-Low'] = Data['high'] - Data['low'] #today's High - Low
    Data['High-Close'] = Data['high'] - Data['close'] #today's High - Close
    Data['High-Open'] = Data['high'] - Data['open'] #today's High - Open
    Data['Close-Open'] = Data['close'] - Data['open'] #today's Close - Open
    Data['Close-High'] = Data['close'] - Data['high'] #today's Close - High
    Data['Close-Low'] = Data['close'] - Data['low'] #today's Close - Low
    Data['Open-Close'] = Data['open'] - Data['close'] #todat's Open - Close
    Data['Open-High'] = Data['open'] - Data['high'] #today's Open - High
    del(Data['open'])
    del(Data['close'])
    del(Data['high'])
    del(Data['low'])
    return Data

def Trader(train_x, train_y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size = 0.5, random_state = 0)
    clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    score = clf.score(X_test, y_test)
    # print(score)
    predict = clf.predict(X_train)
    return clf

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


if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args.training)
    train_X, train_Y = process(training_data)
    trader = Trader(train_X, train_Y)

    handle = 0
    file = open(args.output, 'w', newline='')
    csvCursor = csv.writer(file)
    test_data = load_data(args.testing)
    during = len(test_data)
    for index, row in test_data.iterrows():
            # We will perform your action as the open price in the next day.
            if index == during - 1:
                break
            row = row.reshape((1,4))
            column_names = ['open','high','low','close']
            row = pd.DataFrame(row, columns = column_names)
            test = process_test(row)
            result = trader.predict(test)
            action = get_action(result)
            data = [action]
            csvCursor.writerow(data)
    file.close()