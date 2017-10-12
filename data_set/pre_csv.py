"""
    Do some pre_process for csv file
"""
import pandas as pd
import os

jester_root = '/home/lshi/Database/Jester'
label_csv = os.path.join(jester_root, 'jester-v1-labels.csv')
test_csv = os.path.join(jester_root, 'jester-v1-test.csv')

train_csv = os.path.join(jester_root, 'train.csv')
val_csv = os.path.join(jester_root, 'val.csv')

f1 = pd.read_csv(label_csv, header=None)
f2 = pd.read_csv(test_csv, header=None)
f3 = pd.read_csv(train_csv, header=None)
f4 = pd.read_csv(val_csv, header=None)
print(f3.loc[0, 0])
print(f3.loc[0, 1])
f4.loc[0, 1] = 9224
print(f1)
print(f2)
print(f3)
print(f4)
for i in range(len(f4)):
    num, label = f4.iloc[i]
    for j in range(len(f1)):
        if f1[0][j] == label:
            f4.loc[i, 1] = j
            break
f4.to_csv(os.path.join(jester_root, 'val.csv'), header=None, index=None)

for i in range(len(f3)):
    num, label = f3.iloc[i]
    for j in range(len(f1)):
        if f1[0][j] == label:
            f3.loc[i, 1] = j
            break
f3.to_csv(os.path.join(jester_root, 'train.csv'), header=None, index=None)
