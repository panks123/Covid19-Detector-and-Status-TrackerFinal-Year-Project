import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def split_data(data,ratio):
    np.random.seed(42) #
    shuffled=np.random.permutation(len(data))
    test_data_size=int(len(data)*ratio)
    test_indices=shuffled[:test_data_size]
    train_indices=shuffled[test_data_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__=='__main__':
    df=pd.read_csv('data.csv')
    train,test=split_data(df,0.2)

    x_train=train[['fever','bodyPain','age','runnyNose','diffBreath','visitedAffectedArea','cough']].to_numpy()
    x_test=test[['fever','bodyPain','age','runnyNose','diffBreath','visitedAffectedArea','cough']].to_numpy()

    y_train=train[['infectionProb']].to_numpy().reshape(len(train),)
    y_test=test[['infectionProb']].to_numpy().reshape(len(test),)

    clf=LogisticRegression()
    clf.fit(x_train,y_train)
    # opening a file whwre we will store the model
    file=open('detectorModel.pkl','wb')
    # storing the model into the file
    pickle.dump(clf,file)
    file.close()
