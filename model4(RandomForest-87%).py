import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV

#Бинарный target
df=pd.read_csv("koi.csv",sep=',',comment='#')
df['target']=(df['koi_disposition']=='CONFIRMED').astype(int)

#Расширение списка фич
feature_cols=[
    'koi_score','koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec',
    'koi_period','koi_time0bk','koi_impact','koi_duration',
    'koi_depth','koi_prad','koi_sma','koi_incl','koi_teq',
    'koi_insol','koi_steff','koi_slogg','koi_smet','koi_srad','koi_smass']

#Новые фичи
df['log_period'] = np.log1p(df['koi_period'])  # Лог для длинных периодов
df['depth_over_ror'] = df['koi_depth'] / df['koi_ror']  # Нормализованная глубина
df['habitable_proxy'] = (df['koi_sma'] / (df['koi_teq'] / 255)**2)  # Прокси для habitable zone (Teq~255K для Земли)

feature_cols += ['log_period', 'depth_over_ror', 'habitable_proxy']

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


#Параметры для тюнинга
param_grid={
    'n_estimators': [100,200,300],
    'max_depth': [10,20,None],
    'min_samples_split': [2,5,10],
    'class_weight': ['balanced',None] 
}

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

#Лучшая модель
#GridSearch (5-10 мин)
#grid_search = GridSearchCV(
#    RandomForestClassifier(random_state=42),
#    param_grid,
#    cv=5,
#    scoring='roc_auc',
#    n_jobs=1,
#    verbose=2  # <--- вот здесь
#)
#grid_search.fit(X_train, y_train)

#best_rf=grid_search.best_estimator_
#y_pred=best_rf.predict(X_test)
#print("Доработка на GridSearch")
#print("Best params:", grid_search.best_params_)

#Новая оценка
#print('New Accuracy:',accuracy_score(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))


#Важность признаков
importances=pd.DataFrame({'feature': feature_cols, 'importance': rf_model.feature_importances_})
print(importances.sort_values('importance',ascending=False))

#for cross-validation
scores=cross_val_score(rf_model,X,y,cv=5)
print('CV Score:',scores.mean())

#data visualization
importances.sort_values('importance').plot(kind='barh',x='feature',y='importance')
plt.title('Feature Importance')
plt.show()

# После предсказаний
y_score = rf_model.predict_proba(X_test)[:, 1]
print('ROC-AUC:', roc_auc_score(y_test, y_score))

# Визуализация ROC
RocCurveDisplay.from_predictions(y_test, y_score)
plt.title('ROC Curve')
plt.show()
