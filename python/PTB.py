#!-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys

mail_df = pd.read_csv("./datas/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20 (1).csv")

### 女性向けメールが配信されたデータを削除したデータを作成
male_df = mail_df[mail_df.segment != 'Womens E-Mail'].copy()# 女性向けメールが配信されたデータを削除
male_df["treatment"] = np.where(male_df.segment =="Mens E-Mail",1,0)

# バイアスのあるデータの作成(ここが本では明示的に書かれていなかった。1章のバイアスの作り方を適用)
sample_rules = (male_df.history > 300) | (male_df.recency < 6) | (male_df.channel == 'Multichannel')
biased_df = pd.concat([
    male_df[(sample_rules) & (male_df.treatment == 0)].sample(frac=0.5, random_state=1),
    male_df[(sample_rules) & (male_df.treatment == 1)],
    male_df[(~sample_rules) & (male_df.treatment == 0)],
    male_df[(~sample_rules) & (male_df.treatment == 1)].sample(frac=0.5, random_state=1)
], axis=0, ignore_index=True)

#Zとvisitの相関
Y = biased_df[['treatment']]
X = pd.get_dummies(biased_df[['visit','recency','channel','history']],columns=['channel'],drop_first=True)
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())

#visitを含めた重回帰分析
Y = biased_df[['spend']]
X = pd.get_dummies(biased_df[['treatment','visit','recency','channel','history']],columns=['channel'],drop_first=True)
X = sm.add_constant(X)
results = sm.OLS(Y,X).fit()
table  =results.summary().tables[1]
print(table)

