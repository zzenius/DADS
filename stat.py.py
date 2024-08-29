# 라이브러리 버전 맞춰주기
import subprocess
import sys

with open('requirements.txt', 'w') as f:
    f.write("""
    scikit-learn==1.5.1
    numpy==2.0.0
    joblib==1.4.2    
    pandas==2.2.2
    xgboost==2.1.0
    """)

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# 필요 라이브러리 import
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# test set
columns = [
    'hour', 'ID', 'long', 'lat', 'day', 'month', 'year', 'dow', 'woy', 
    'weekend', 'line', 'peak', 'precipitation', 'precipitation2', 'baseball'
]
df_test = pd.read_csv('df_test_final.csv')
X_test = df_test[columns]

# 모델 load
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# 예측값 구하기
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf_xgb = np.round(0.2 * np.array(y_pred_rf) + 0.8 * np.array(y_pred_xgb))

# 예측값 result V8에 대입
result = pd.read_csv('result.csv', encoding = 'cp949')
result.loc[:, 'V8'] = y_pred_rf_xgb

# 음수값 0으로 대체
def func(df_data):
    count = df_data['V8']
    return max(count, 0)

result['V8'] = result.apply(func, axis = 1)

# 바다축제날 추가 인원 더해주기
result.loc[(result['V5']=='다대포해수욕장') & (result['V2']==20240726) & (result['V3'] == 18), 'V8'] += 2660
result.loc[(result['V5']=='다대포해수욕장') & (result['V2']==20240726) & (result['V3'] == 19), 'V8'] += 2914
result.loc[(result['V5']=='다대포해수욕장') & (result['V2']==20240726) & (result['V3'] == 20), 'V8'] += 5698
result.loc[(result['V5']=='다대포해수욕장') & (result['V2']==20240726) & (result['V3'] == 21), 'V8'] += 2506
result.loc[(result['V5']=='다대포해수욕장') & (result['V2']==20240726) & (result['V3'] == 22), 'V8'] += 691

# result data 저장
result.to_csv('result_final.csv', encoding = 'cp949')