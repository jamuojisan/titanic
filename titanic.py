import sys
import os
import datetime
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np

# DIR
DIR_HOME = os.getcwd() + os.sep + ".."
DIR_DATA = DIR_HOME + os.sep + "data"

def main():
    #データのロード
    train, test = data_load(DIR_DATA)
    #学習データを特徴量と目的変数に分ける
    train_x = train.drop(["Survived"], axis=1)
    train_y = train["Survived"]

    #テストデータは特徴量のみなので、そのまま
    test_x = test.copy()

    #前処理
    train_x, test_x= preprocessing(train_x, test_x)

    #モデルの作成と学習
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(train_x, train_y)
    print(model)

    #予測値を出す
    pred = model.predict_proba(test_x)[:, 1]
    
    #テストデータの予測値をにちに変換
    pred_label = np.where(pred > 0.5, 1, 0)

    #提出用ファイルの作成
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred_label})
    submission.to_csv(DIR_HOME + os.sep + "submission_first.csv", index = False)

def preprocessing(train_x, test_x):
    #乗客IDの削除
    train_x = train_x.drop(["PassengerId"], axis=1)
    test_x = test_x.drop(["PassengerId"], axis=1)

    #名前、ticket, cabinの削除
    train_x = train_x.drop(["Name", "Ticket", "Cabin"], axis=1)
    test_x = test_x.drop(["Name", "Ticket", "Cabin"], axis=1)

    #GBDTには変数に文字列をいれれないため、文字列を数字にencode
    for c in ["Sex", "Embarked"]:
        #学習データに基づいてエンコード内容を決める
        le = LabelEncoder()
        le.fit(train_x[c].fillna("Na"))
        #学習データ、テストデータを変換
        train_x[c] = le.transform(train_x[c].fillna("Na"))
        test_x[c] = le.transform(test_x[c].fillna("Na"))

    return train_x, test_x
    

def data_load(dir):
    train = pd.read_csv(dir + os.sep + "train.csv")
    test = pd.read_csv(dir + os.sep + "test.csv")

    return train, test


if __name__ == '__main__':
    main() 
