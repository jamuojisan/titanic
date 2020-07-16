import sys
import os
import datetime
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

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

    #各foldのスコアを保存するリスト
    scores_accuracy = []
    scores_logloss = []

    #クロスバリデーション
    #学習データを４分割、うち１つをバリデーションとすることを、四回繰り返す。

    kf = KFold(n_splits=4, shuffle = True, random_state= 71)
    for tr_idx, va_idx, in kf.split(train_x):
        #学習データを学習データとバリデーションデータに分割
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        #学習
        model = XGBClassifier(n_estimators=20, random_state=71)
        model.fit(tr_x, tr_y)

        #バリデーションデータの予測値を確率で出力
        va_pred = model.predict_proba(va_x)[:, 1]

        #バリデーションでのスコア
        logloss = log_loss(va_y, va_pred)
        accuracy = accuracy_score(va_y, va_pred > 0.5)

        #スコアの保存
        scores_accuracy.append(accuracy)
        scores_logloss.append(logloss)
    #各スコアの平均
    logloss = np.mean(scores_logloss)
    accuracy = np.mean(scores_accuracy)
    print(f"logloss: {logloss:.4f}, accuracy: {accuracy:.4f}")
    exit()
  

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
