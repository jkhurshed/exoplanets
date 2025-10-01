import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Бинарный target
df=pd.read_csv("koi.csv",sep=',',comment='#')
df['target']=(df['koi_disposition']=='CONFIRMED').astype(int)
feature_cols=[
    'koi_period','koi_time0bk','koi_impact','koi_duration',
    'koi_depth','koi_prad','koi_sma','koi_incl','koi_teq',
    'koi_insol','koi_steff','koi_slogg','koi_smet','koi_srad','koi_smass']

#Удаление строк с пропусками в target или ключевых фичах
df=df.dropna(subset=['target']+feature_cols)

X=df[feature_cols] #признаки
y=df['target'] #target

#Заполнение пропусков с медианой
X=X.fillna(X.median())

#Разделение
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
print(f'Train: {X_train.shape},Test: {X_test.shape}')

#Cоздание и обучение модели
rf_model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
rf_model.fit(X_train, y_train)

#Предсказания
y_pred=rf_model.predict(X_test)

#Оценка
print('Accuracy:',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))

#Важность признаков
importances=pd.DataFrame({'feature': feature_cols, 'importance': rf_model.feature_importances_})
print(importances.sort_values('importance',ascending=False))


