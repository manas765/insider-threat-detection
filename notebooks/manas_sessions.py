import pandas as pd

path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\logon.csv"

df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['user', 'pc', 'date'])

# Shift the next row's date and activity into each row, so we can compare
df['next_date'] = df.groupby(['user', 'pc'])['date'].shift(-1)
df['next_activity'] = df.groupby(['user', 'pc'])['activity'].shift(-1)

# Only keep rows where a Logon is immediately followed by a Logoff
sessions = df[(df['activity'] == 'Logon') & (df['next_activity'] == 'Logoff')].copy()
sessions['session_duration_mins'] = (sessions['next_date'] - sessions['date']).dt.total_seconds() / 60

print(sessions[['user', 'pc', 'date', 'next_date', 'session_duration_mins']].head(10))
print("\nHow many valid sessions found:", len(sessions))
print("\nSession duration stats:")
print(sessions['session_duration_mins'].describe())
sessions['login_hour'] = sessions['date'].dt.hour
sessions['after_hours_flag'] = sessions['login_hour'].apply(lambda h: 1 if (h < 6 or h >= 18) else 0)

print("\nLogin hour distribution:")
print(sessions['login_hour'].value_counts().sort_index())

print("\nAfter-hours flag counts:")
print(sessions['after_hours_flag'].value_counts())
sessions['day'] = sessions['date'].dt.date

daily = sessions.groupby(['user', 'day']).agg(
    login_hour=('login_hour', 'first'),
    after_hours_flag=('after_hours_flag', 'max'),
    session_duration_mins=('session_duration_mins', 'sum')
).reset_index()

print("\nDaily aggregated logon data:")
print(daily.head(10))
print("\nShape:", daily.shape)
device_path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\device.csv"
device_df = pd.read_csv(device_path)
device_df['date'] = pd.to_datetime(device_df['date'])
device_df['day'] = device_df['date'].dt.date

usb_daily = device_df[device_df['activity'] == 'Connect'].groupby(['user', 'day']).size().reset_index(name='usb_events_count')

print("\nDaily USB connect counts:")
print(usb_daily.head(10))
print("\nShape:", usb_daily.shape)
merged = daily.merge(usb_daily, on=['user', 'day'], how='left')
merged['usb_events_count'] = merged['usb_events_count'].fillna(0).astype(int)

print("\nMerged daily table (logon + usb):")
print(merged.head(10))
print("\nShape:", merged.shape)
print("\nUSB count stats:")
print(merged['usb_events_count'].describe())
file_path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\file.csv"
file_df = pd.read_csv(file_path)
file_df['date'] = pd.to_datetime(file_df['date'])
file_df['day'] = file_df['date'].dt.date

files_daily = file_df.groupby(['user', 'day']).size().reset_index(name='files_accessed_count')

print("\nDaily file access counts:")
print(files_daily.head(10))
print("\nShape:", files_daily.shape)

merged = merged.merge(files_daily, on=['user', 'day'], how='left')
merged['files_accessed_count'] = merged['files_accessed_count'].fillna(0).astype(int)

print("\nMerged table now (logon + usb + file):")
print(merged.head(10))
print("\nShape:", merged.shape)
print("\nFiles accessed stats:")
print(merged['files_accessed_count'].describe())
email_path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\email.csv"
email_df = pd.read_csv(email_path)
email_df['date'] = pd.to_datetime(email_df['date'])
email_df['day'] = email_df['date'].dt.date

email_daily = email_df.groupby(['user', 'day']).size().reset_index(name='email_count')

print("\nDaily email counts:")
print(email_daily.head(10))
print("\nShape:", email_daily.shape)

merged = merged.merge(email_daily, on=['user', 'day'], how='left')
merged['email_count'] = merged['email_count'].fillna(0).astype(int)

print("\nMerged table now (logon + usb + file + email):")
print(merged.head(10))
print("\nShape:", merged.shape)
insiders_path = r"C:\Users\manas\Downloads\answers\answers\insiders.csv"
insiders = pd.read_csv(insiders_path)

# Only keep r4.2 insiders
insiders_42 = insiders[insiders['dataset'] == 4.2].copy()
insiders_42['start'] = pd.to_datetime(insiders_42['start'])
insiders_42['end'] = pd.to_datetime(insiders_42['end'])

print("\nMalicious users in r4.2:")
print(insiders_42[['user', 'start', 'end']])
merged['day'] = pd.to_datetime(merged['day'])
merged['is_malicious'] = 0

for _, row in insiders_42.iterrows():
    mask = (
        (merged['user'] == row['user']) &
        (merged['day'] >= row['start'].normalize()) &
        (merged['day'] <= row['end'].normalize())
    )
    merged.loc[mask, 'is_malicious'] = 1

print("\nLabel distribution:")
print(merged['is_malicious'].value_counts())
print("\nPercentage malicious:", (merged['is_malicious'].sum() / len(merged)) * 100, "%")
merged = merged.sort_values('day').reset_index(drop=True)

split_index = int(len(merged) * 0.8)
train = merged.iloc[:split_index]
test = merged.iloc[split_index:]

print("Train date range:", train['day'].min(), "to", train['day'].max())
print("Test date range:", test['day'].min(), "to", test['day'].max())
print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain malicious count:", train['is_malicious'].sum())
print("Test malicious count:", test['is_malicious'].sum())
from imblearn.over_sampling import SMOTE

feature_cols = ['login_hour', 'after_hours_flag', 'session_duration_mins',
                 'usb_events_count', 'files_accessed_count', 'email_count']

X_train = train[feature_cols]
y_train = train['is_malicious']
X_test = test[feature_cols]
y_test = test['is_malicious']

print("\nBefore SMOTE:", y_train.value_counts().to_dict())

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_resampled.value_counts().to_dict())
import os
os.makedirs('data/processed', exist_ok=True)

X_train_resampled.to_csv('data/processed/X_train.csv', index=False)
y_train_resampled.to_csv('data/processed/y_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("\nExported successfully!")
print("X_train:", X_train_resampled.shape)
print("X_test:", X_test.shape)