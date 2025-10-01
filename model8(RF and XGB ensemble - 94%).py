import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Загрузка данных
df = pd.read_csv("koi.csv", sep=',', comment='#')
df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)

# Расширенный список фич (оригинальные + новые SNR-метрики)
feature_cols = [
    'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq',
    'koi_steff', 'koi_slogg', 'koi_smet', 'koi_srad', 'koi_smass',
    'koi_model_snr', 'koi_max_sngle_ev', 'koi_max_mult_ev'  # Новые SNR-фичи
]

# Новые фичи (с обработкой деления на ноль)
df['depth_over_ror'] = np.where(df['koi_ror'] != 0, df['koi_depth'] / df['koi_ror'], np.nan)  # Избежать inf
df['habitable_proxy'] = np.where(df['koi_teq'] != 0, df['koi_sma'] / (df['koi_teq'] / 255)**2, np.nan)  # Избежать inf
df['period_over_depth'] = np.where(df['koi_depth'] != 0, df['koi_period'] / df['koi_depth'], np.nan)  # Избежать inf

feature_cols += ['depth_over_ror', 'habitable_proxy', 'period_over_depth']

# Удаление строк с пропусками в target или ключевых фичах
df = df.dropna(subset=['target'] + feature_cols)

X = df[feature_cols]  # Признаки
y = df['target']  # Target

# Заполнение пропусков медианой (после создания фич, чтобы обработать nan от where)
X = X.fillna(X.median())

# Обработка inf и очень больших значений (clipping для стабильности)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())  # Повторное заполнение после replace
X = np.clip(X, -1e10, 1e10)  # Clip экстремальных значений

# Корреляция: Удаление коррелированных (>0.95)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)
feature_cols = [f for f in feature_cols if f not in to_drop]

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# Проверка распределения классов
print(f'Majority class (0) in train: {len(y_train[y_train == 0])}')
print(f'Minority class (1) in train: {len(y_train[y_train == 1])}')

# Балансировка: Только SMOTE (без undersampling, чтобы избежать ошибки с ratio)
smote = SMOTE(sampling_strategy=1.0, random_state=42)  # Равные классы
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f'After SMOTE - Majority: {len(y_train_bal[y_train_bal == 0])}, Minority: {len(y_train_bal[y_train_bal == 1])}')

# Упрощённый тюнинг XGBoost с n_jobs=1 (чтобы избежать memory/parallel ошибок)
param_grid = {
    'n_estimators': [300],
    'max_depth': [4, 6],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'scale_pos_weight': [1.0]  # Поскольку балансировали SMOTE, scale_pos=1
}
xgb_base = XGBClassifier(random_state=42)
xgb_tuned = GridSearchCV(xgb_base, param_grid, cv=3, scoring='f1_macro', n_jobs=1)  # n_jobs=1, cv=3 для скорости
xgb_tuned.fit(X_train_bal, y_train_bal)  # На balanced данных
best_xgb = xgb_tuned.best_estimator_
print('Best XGBoost Params:', xgb_tuned.best_params_)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf.fit(X_train_bal, y_train_bal)

# Stacking ансамбль (улучшенный)
stacking = StackingClassifier(
    estimators=[('xgb', best_xgb), ('rf', rf)], 
    final_estimator=LogisticRegression(random_state=42), 
    cv=3  # Уменьшили cv для скорости
)
stacking.fit(X_train_bal, y_train_bal)

# Предсказания на лучшей модели (Stacking)
y_pred = stacking.predict(X_test)
y_score = stacking.predict_proba(X_test)[:, 1]

# Оценка
print('Stacking Accuracy:', accuracy_score(y_test, y_pred))
print('Stacking F1 Macro:', f1_score(y_test, y_pred, average='macro'))
print('Stacking ROC-AUC:', roc_auc_score(y_test, y_score))
print(classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Важность признаков (от XGBoost)
importances = pd.DataFrame({'feature': feature_cols, 'importance': best_xgb.feature_importances_})
print(importances.sort_values('importance', ascending=False))

# Cross-validation на полной модели (с n_jobs=1 для стабильности)
cv_scores = cross_val_score(stacking, X, y, cv=3, scoring='roc_auc', n_jobs=1)
print('CV ROC-AUC:', cv_scores.mean())

# Визуализация
importances.sort_values('importance').plot(kind='barh', x='feature', y='importance')
plt.title('Feature Importance (XGBoost)')
plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_score)
plt.title('ROC Curve (Stacking)')
plt.show()
