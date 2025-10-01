import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from xgboost import XGBClassifier

#Бинарный target
df=pd.read_csv("koi.csv",sep=',',comment='#')
df['target']=(df['koi_disposition']=='CONFIRMED').astype(int)

#Расширение списка фич
feature_cols=[
    'koi_score','koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec',
    'koi_period','koi_time0bk','koi_impact','koi_duration',
    'koi_depth','koi_prad','koi_sma','koi_incl','koi_teq',
    'koi_steff','koi_slogg','koi_smet','koi_srad','koi_smass']

#Новые фичи
#df['log_period'] = np.log1p(df['koi_period'])  # Лог для длинных периодов. Дропнул, не даёт никакого результата
df['depth_over_ror'] = df['koi_depth'] / df['koi_ror']  # Нормализованная глубина
df['habitable_proxy'] = (df['koi_sma'] / (df['koi_teq'] / 255)**2)  # Прокси для habitable zone (Teq~255K для Земли)

feature_cols += ['depth_over_ror', 'habitable_proxy']

#Удаление строк с пропусками в target или ключевых фичах
df=df.dropna(subset=['target']+feature_cols)

X=df[feature_cols] #признаки
y=df['target'] #target

#Заполнение пропусков с медианой
X=X.fillna(X.median())


#Корреляция: Удаление коррелированных (>0.95)
corr_matrix=X.corr().abs()
upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
to_drop=[column for column in upper.columns if any(upper[column]>0.95)]
X=X.drop(columns=to_drop)
feature_cols=[f for f in feature_cols if f not in to_drop]



#Разделение
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
print(f'Train: {X_train.shape},Test: {X_test.shape}')


#Cоздание и обучение модели
scale_pos = sum(y_train == 0) / sum(y_train == 1)
xgb=XGBClassifier(n_estimators=200, max_depth=10, random_state=42, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
xgb.fit(X_train, y_train)

#Предсказания
y_pred=xgb.predict(X_test)


#Оценка
print('Accuracy:',accuracy_score(y_test,y_pred))
print("XGBoost Accuracy", accuracy_score(y_test, xgb.predict(X_test)))
print(classification_report(y_test,y_pred))
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))


#Важность признаков
importances=pd.DataFrame({'feature': feature_cols, 'importance': xgb.feature_importances_})
print(importances.sort_values('importance',ascending=False))

#for cross-validation
scores=cross_val_score(xgb,X,y,cv=5)
print('CV Score:',scores.mean())

#data visualization
importances.sort_values('importance').plot(kind='barh',x='feature',y='importance')
plt.title('Feature Importance')
plt.show()

# После предсказаний
y_score =xgb.predict_proba(X_test)[:, 1]
print('ROC-AUC:', roc_auc_score(y_test, y_score))

# Визуализация ROC
RocCurveDisplay.from_predictions(y_test, y_score)
plt.title('ROC Curve')
plt.show()
