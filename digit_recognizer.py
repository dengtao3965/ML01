import pandas as pd
import numpy as np
import os


def save_result(y_pred, file_name):
    result_df = pd.DataFrame({'ImageId': range(1, len(y_pred) + 1), 'Label': y_pred})
    result_df.to_csv(file_name, index=False)

def init():
    data=pd.read_table("data/train.csv",header=0,sep=',')
    train_x=np.array(data.iloc[:,1:]).astype('float32')
    train_y=np.array(data.iloc[:,0])
    train_x/=255
    data=pd.read_table("data/test.csv",header=0,sep=',')
    test_x=np.array(data.iloc[:,:]).astype('float32')

    return train_x,train_y,test_x


train_x,train_y,test_x=init()
save_file=os.path.join('result','random_forest.csv')
from sklearn.ensemble import RandomForestClassifier
clf =RandomForestClassifier(n_estimators=150)
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)

save_result(y_pred,save_file)
