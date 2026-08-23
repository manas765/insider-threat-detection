import pandas as pd
import os
from imblearn.over_sampling import SMOTE

# ---------- 1. LOGON DATA ----------
path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\logon.csv"
df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['user', 'pc', 'date'])

df['next_date'] = df.groupby(['user', 'pc'])['date'].shift(-1)
df['next_activity'] = df.groupby(['user', 'pc'])['activity'].shift(-1)

sessions = df[(df['activity'] == 'Logon') & (df['next_activity'] == 'Logoff')].copy()
sessions['session_duration_mins'] = (sessions['next_date'] - sessions['date']).dt.total_seconds() / 60
sessions['login_hour'] = sessions['date'].dt.hour
sessions['after_hours_flag'] = sessions['login_hour'].apply(lambda h: 1 if (h < 6 or h >= 18) else 0)
sessions['day'] = sessions['date'].dt.date

daily = sessions.groupby(['user', 'day']).agg(
    login_hour=('login_hour', 'first'),
    after_hours_flag=('after_hours_flag', 'max'),
    session_duration_mins=('session_duration_mins', 'sum')
).reset_index()

# ---------- 2. USB CONNECT EVENTS ----------
device_path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\device.csv"
device_df = pd.read_csv(device_path)
device_df['date'] = pd.to_datetime(device_df['date'])
device_df['day'] = device_df['date'].dt.date

usb_daily = device_df[device_df['activity'] == 'Connect'].groupby(['user', 'day']).size().reset_index(name='usb_events_count')
merged = daily.merge(usb_daily, on=['user', 'day'], how='left')
merged['usb_events_count'] = merged['usb_events_count'].fillna(0).astype(int)

# ---------- 3. FILE ACCESS ----------
file_path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\file.csv"
file_df = pd.read_csv(file_path)
file_df['date'] = pd.to_datetime(file_df['date'])
file_df['day'] = file_df['date'].dt.date

files_daily = file_df.groupby(['user', 'day']).size().reset_index(name='files_accessed_count')
merged = merged.merge(files_daily, on=['user', 'day'], how='left')
merged['files_accessed_count'] = merged['files_accessed_count'].fillna(0).astype(int)

# ---------- 4. EMAIL COUNT ----------
email_path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\email.csv"
email_df = pd.read_csv(email_path)
email_df['date'] = pd.to_datetime(email_df['date'])
email_df['day'] = email_df['date'].dt.date

email_daily = email_df.groupby(['user', 'day']).size().reset_index(name='email_count')
merged = merged.merge(email_daily, on=['user', 'day'], how='left')
merged['email_count'] = merged['email_count'].fillna(0).astype(int)

# ---------- 5. HTTP UNIQUE DOMAINS ----------
http_path = r"C:\Users\manas\Downloads\r4.2.tar(1)\r4.2(1)\r4.2\http.csv"
domains_list = []
chunksize = 500000
for chunk in pd.read_csv(http_path, usecols=['date', 'user', 'url'], chunksize=chunksize, engine='python', on_bad_lines='skip'):
    chunk['date'] = pd.to_datetime(chunk['date'])
    chunk['day'] = chunk['date'].dt.date
    chunk['domain'] = chunk['url'].str.extract(r'://([^/]+)/')
    domains_list.append(chunk[['user', 'day', 'domain']])

http_df = pd.concat(domains_list, ignore_index=True)
domains_daily = http_df.groupby(['user', 'day'])['domain'].nunique().reset_index(name='unique_domains_visited')

merged['day'] = merged['day']  # already date type
merged = merged.merge(domains_daily, on=['user', 'day'], how='left')
merged['unique_domains_visited'] = merged['unique_domains_visited'].fillna(0).astype(int)

# ---------- 6. EXTERNAL EMAIL RECIPIENTS ----------
def count_external_recipients(to_field):
    if pd.isna(to_field):
        return 0
    recipients = to_field.split(';')
    return sum(1 for r in recipients if 'dtaa.com' not in r.lower())

email_df['ext_recipient_count'] = email_df['to'].apply(count_external_recipients)
ext_daily = email_df.groupby(['user', 'day'])['ext_recipient_count'].sum().reset_index(name='email_ext_recipient_count')

merged = merged.merge(ext_daily, on=['user', 'day'], how='left')
merged['email_ext_recipient_count'] = merged['email_ext_recipient_count'].fillna(0).astype(int)

# ---------- 7. FILE COPY TO REMOVABLE (USB windows + file access) ----------
device_sorted = device_df.sort_values(['user', 'pc', 'date'])
device_sorted['next_date'] = device_sorted.groupby(['user', 'pc'])['date'].shift(-1)
device_sorted['next_activity'] = device_sorted.groupby(['user', 'pc'])['activity'].shift(-1)

usb_windows = device_sorted[
    (device_sorted['activity'] == 'Connect') &
    (device_sorted['next_activity'] == 'Disconnect')
][['user', 'pc', 'date', 'next_date']].rename(columns={'date': 'usb_start', 'next_date': 'usb_end'})

file_df_sorted = file_df.sort_values(['user', 'pc', 'date'])
usb_windows_sorted = usb_windows.sort_values(['user', 'pc', 'usb_start'])

matched_rows = []
for (user, pc), group in usb_windows_sorted.groupby(['user', 'pc']):
    user_files = file_df_sorted[(file_df_sorted['user'] == user) & (file_df_sorted['pc'] == pc)]
    if user_files.empty:
        continue
    for _, window in group.iterrows():
        matches = user_files[(user_files['date'] >= window['usb_start']) & (user_files['date'] <= window['usb_end'])]
        if not matches.empty:
            matched_rows.append(matches)

if matched_rows:
    file_usb_matched = pd.concat(matched_rows, ignore_index=True)
else:
    file_usb_matched = pd.DataFrame(columns=file_df.columns)

file_usb_matched['day'] = pd.to_datetime(file_usb_matched['date']).dt.date
file_copy_daily = file_usb_matched.groupby(['user', 'day']).size().reset_index(name='file_copy_to_removable')

merged = merged.merge(file_copy_daily, on=['user', 'day'], how='left')
merged['file_copy_to_removable'] = merged['file_copy_to_removable'].fillna(0).astype(int)

# ---------- 8. LABEL WITH insiders.csv ----------
insiders_path = r"C:\Users\manas\Downloads\answers\answers\insiders.csv"
insiders = pd.read_csv(insiders_path)
insiders_42 = insiders[insiders['dataset'] == 4.2].copy()
insiders_42['start'] = pd.to_datetime(insiders_42['start'])
insiders_42['end'] = pd.to_datetime(insiders_42['end'])

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

# ---------- 9. TIME-BASED SPLIT ----------
merged = merged.sort_values('day').reset_index(drop=True)
split_index = int(len(merged) * 0.8)
train = merged.iloc[:split_index]
test = merged.iloc[split_index:]

feature_cols = ['login_hour', 'after_hours_flag', 'session_duration_mins',
                 'usb_events_count', 'files_accessed_count', 'email_count',
                 'unique_domains_visited', 'email_ext_recipient_count',
                 'file_copy_to_removable']

X_train = train[feature_cols]
y_train = train['is_malicious']
X_test = test[feature_cols]
y_test = test['is_malicious']

print("\nBefore SMOTE:", y_train.value_counts().to_dict())
print("file_copy_to_removable stats (train):")
print(X_train['file_copy_to_removable'].describe())

# ---------- 10. EXPORTS ----------
os.makedirs('data/processed', exist_ok=True)

# Raw (real, imbalanced) — for models that want true data
X_train.to_csv('data/processed/X_train_raw.csv', index=False)
y_train.to_csv('data/processed/y_train_raw.csv', index=False)

# Benign-only — for autoencoder / one-class models
X_train_benign = X_train[y_train == 0]
X_train_benign.to_csv('data/processed/X_train_benign.csv', index=False)

# SMOTE'd — kept only for a supervised baseline if needed
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_resampled.to_csv('data/processed/X_train_smote.csv', index=False)
y_train_resampled.to_csv('data/processed/y_train_smote.csv', index=False)

# Test set — always real, never resampled
X_test.to_csv('data/processed/X_test.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("\nExported all versions:")
print("Raw train:", X_train.shape, "| Benign-only train:", X_train_benign.shape,
      "| SMOTE'd train:", X_train_resampled.shape, "| Test:", X_test.shape)
print("Feature columns (9):", feature_cols)