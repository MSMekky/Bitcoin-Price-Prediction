# Bitcoin Price Prediction - Master’s Thesis Implementation

# Author: Mohamed Mekky (GH1040715)
# Program: MSc Data Science, AI and Digital Business

# Date: January 2026

# Version: v.7.3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import shap
import yfinance as yf
from datetime import datetime, timedelta
from collections import Counter
import warnings

# Configuration

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

start_time = datetime.now()

CONFIG = {
    'seq_len': 10,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'rf_n_estimators': 100,
    'rf_max_depth': 10,
    'gru_units_1': 64,
    'gru_units_2': 32,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'patience': 10,
    'cost': 0.002,
    'target_sharpe': 0.30
}

# Output Setup
os.makedirs('figures', exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print(f"BITCOIN PRICE PREDICTION - FINAL THESIS EXECUTION")
print(f"Execution Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# 1. DATA ACQUISITION & MERGING

print("\n>>> STEP 1: DATA ACQUISITION & MERGING")

BTC_FILE = 'BTC-USD.csv'
START_DATE = '2018-01-01'
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

# 1.1 Bitcoin Data

if not os.path.exists(BTC_FILE):
    print(f"   Downloading Bitcoin data ({START_DATE} to {END_DATE})…")
    try:
        btc = yf.download('BTC-USD', start=START_DATE, end=END_DATE, progress=False)
        btc.reset_index(inplace=True)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)
        btc.to_csv(BTC_FILE, index=False)
    except Exception as e:
        print(f"    Error downloading Bitcoin: {e}")
        exit(1)
else:
    btc = pd.read_csv(BTC_FILE)

btc['Date'] = pd.to_datetime(btc['Date'])
btc = btc.sort_values('Date').reset_index(drop=True)
print(f"    Bitcoin data ready: {len(btc)} rows")

# 1.2 Google Trends (Multi-File Merge)

trend_files = glob.glob('multiTimeline*.csv')
if not trend_files:
    print("    CRITICAL ERROR: No 'multiTimeline' files found!")
    print("      Please upload your Google Trends CSV files.")
    exit(1)

print(f"    Found {len(trend_files)} Trends files. Merging…")

dfs = []
for f in trend_files:
    try:
        df = pd.read_csv(f, header=1)
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'].replace('<1', '0'), errors='coerce')
        df.set_index('date', inplace=True)
        df.rename(columns={'value': f"trend_{trend_files.index(f)}"}, inplace=True)
        dfs.append(df)
    except:
        pass

# Merge & Upsample

full_trends = pd.concat(dfs, axis=1)
full_trends['trends'] = full_trends.mean(axis=1) # Composite Index
trends_monthly = full_trends[['trends']].sort_index()

# Reindex to Daily

trends_daily = trends_monthly.reindex(btc['Date'])
trends_daily['trends'] = trends_daily['trends'].interpolate(method='pchip').ffill().bfill()
trends_daily = trends_daily.reset_index().rename(columns={'Date': 'date'})

# Final Merge

btc.rename(columns={'Date': 'date'}, inplace=True)
data = pd.merge(btc[['date', 'Close']], trends_daily[['date', 'trends']], on='date', how='inner')
print(f"    Final Merged Dataset: {len(data)} observations (Real Data)")

# 2. FEATURE ENGINEERING

print("\n>>> STEP 2: FEATURE ENGINEERING")
data['ret'] = data['Close'].pct_change()

for i in [1, 2, 3, 5, 10]:
    data[f'ret{i}'] = data['ret'].shift(i)

data['vol7'] = data['ret'].shift(1).rolling(7).std()
data['vol14'] = data['ret'].shift(1).rolling(14).std()
data['tr1'] = data['trends'].shift(1)
data['tr_diff'] = data['trends'].diff()
data['tr_ma'] = data['trends'].rolling(7).mean()

data = data.dropna().reset_index(drop=True)
feat_cols = ['ret1', 'ret2', 'ret3', 'ret5', 'ret10',
             'vol7', 'vol14',
             'trends', 'tr1', 'tr_diff', 'tr_ma']
target_col = 'ret'

X = data[feat_cols].values
y = data[target_col].values
print(f"    Features created: {len(feat_cols)}")

# 3. SPLITTING & SCALING

print("\n>>> STEP 3: PREPROCESSING")
n_total = len(X)
n_train = int(CONFIG['train_ratio'] * n_total)
n_val = int(CONFIG['val_ratio'] * n_total)
n_test = n_total - n_train - n_val

print(f"   Total Samples: {n_total}")
print(f"   Training:      {n_train} ({n_train/n_total*100:.1f}%)")
print(f"   Validation:    {n_val} ({n_val/n_total*100:.1f}%)")
print(f"   Test:          {n_test} ({n_test/n_total*100:.1f}%)")

Xtr, ytr = X[:n_train], y[:n_train]
Xva, yva = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
Xte, yte = X[n_train+n_val:], y[n_train+n_val:]

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xva = scaler.transform(Xva)
Xte = scaler.transform(Xte)

# 4. MODELS TRAINING

print("\n>>> STEP 4: TRAINING MODELS")

# RF

rf = RandomForestRegressor(n_estimators=CONFIG['rf_n_estimators'], max_depth=CONFIG['rf_max_depth'], random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(Xtr, ytr)
rf_pred_test = rf.predict(Xte)
print(f"    Random Forest trained")

# GRU

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

Xtr_seq, ytr_seq = create_sequences(Xtr, ytr, CONFIG['seq_len'])
Xva_seq, yva_seq = create_sequences(Xva, yva, CONFIG['seq_len'])
Xte_seq, yte_seq = create_sequences(Xte, yte, CONFIG['seq_len'])

model = keras.Sequential([
    layers.Input(shape=(CONFIG['seq_len'], len(feat_cols))),
    layers.GRU(CONFIG['gru_units_1'], return_sequences=True), layers.Dropout(CONFIG['dropout_rate']),
    layers.GRU(CONFIG['gru_units_2']), layers.Dropout(CONFIG['dropout_rate']),
    layers.Dense(1)
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss='mse')

history = model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq), epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
                    callbacks=[EarlyStopping(monitor='val_loss', patience=CONFIG['patience'], restore_best_weights=True)], verbose=0)
gru_pred_test = model.predict(Xte_seq, verbose=0).flatten()
print(f"    GRU trained")

# 5.  EVALUATION & REPORTING

print("\n>>> STEP 5: EVALUATION & REPORTING")

rf_pred_aligned = rf_pred_test[CONFIG['seq_len']:]
yte_aligned = yte[CONFIG['seq_len']:]

def calc_metrics_full(y_pred, y_true, cost):
    signals = np.sign(y_pred)
    raw_ret = signals * y_true
    trades = np.abs(np.diff(signals, prepend=signals[0]))
    net_ret = raw_ret - (trades * cost)
    sharpe = (np.mean(net_ret)/np.std(net_ret)*np.sqrt(252)) if np.std(net_ret)>0 else 0

    n_trades = int(np.sum(trades > 0))
    total_return = np.sum(net_ret)
    win_rate = np.mean(net_ret > 0)

    return {'sharpe': sharpe, 'net_returns': net_ret, 'n_trades': n_trades, 'win_rate': win_rate, 'total_return': total_return}

rf_metrics = calc_metrics_full(rf_pred_aligned, yte_aligned, CONFIG['cost'])
gru_metrics = calc_metrics_full(gru_pred_test, yte_aligned, CONFIG['cost'])
bh_sharpe = (np.mean(yte_aligned)/np.std(yte_aligned)*np.sqrt(252))

rf_da = np.mean(np.sign(yte_aligned) == np.sign(rf_pred_aligned))
gru_da = np.mean(np.sign(yte_aligned) == np.sign(gru_pred_test))
t_stat, p_val = stats.ttest_rel((np.sign(yte_aligned)==np.sign(gru_pred_test)).astype(int),
                                (np.sign(yte_aligned)==np.sign(rf_pred_aligned)).astype(int))

q_size = len(yte_aligned) // 4
rf_q, gru_q = [], []
for i in range(4):
    s, e = i*q_size, (i+1)*q_size if i<3 else len(yte_aligned)
    rf_q.append(np.mean(np.sign(yte_aligned[s:e]) == np.sign(rf_pred_aligned[s:e])))
    gru_q.append(np.mean(np.sign(yte_aligned[s:e]) == np.sign(gru_pred_test[s:e])))

rf_cv = np.std(rf_q)/np.mean(rf_q) if np.mean(rf_q) != 0 else 0
gru_cv = np.std(gru_q)/np.mean(gru_q) if np.mean(gru_q) != 0 else 0

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(Xte)
shap_imp = np.abs(shap_values).mean(axis=0)
feat_imp = pd.DataFrame({'Feature': feat_cols, 'Imp': shap_imp}).sort_values('Imp', ascending=False)

quarterly_top3 = []
all_top_features = []
for q_idx in range(4):
    s = q_idx * q_size
    e = (q_idx+1) * q_size if q_idx < 3 else len(Xte)
    shap_q = np.abs(shap_values[s:e]).mean(axis=0)
    top3_indices = np.argsort(shap_q)[-3:][::-1]
    top3 = [feat_cols[idx] for idx in top3_indices]
    quarterly_top3.append(top3)
    all_top_features.extend(top3)

feature_frequency = Counter(all_top_features)
stable_count = sum(1 for count in feature_frequency.values() if count >= 3)
behavioral_in_top5 = sum(1 for f in feat_imp.head(5)['Feature'] if 'tr' in f or f == 'trends')

# FINAL THESIS REPORT 

print("\n" + "="*80)
print("FINAL THESIS REPORT - METRICS FOR CHAPTER 4")
print("="*80)

print(f"\n[RQ1] DIRECTIONAL ACCURACY")
print(f"Random Forest:  {rf_da:.2%}")
print(f"GRU Model:      {gru_da:.2%}")
print(f"Improvement:    {(gru_da-rf_da)*100:+.2f} percentage points")
print(f"T-test p-value: {p_val:.4f} ({'Significant' if p_val<0.05 else 'Not significant'} at alpha=0.05)")

print(f"\n[RQ2] PREDICTIVE STABILITY (CV)")
print(f"Random Forest:  {rf_cv:.4f}")
print(f"GRU Model:      {gru_cv:.4f}")
print(f"Quarterly Accuracy Breakdown:")
for i, (r, g) in enumerate(zip(rf_q, gru_q)):
    print(f"  Q{i+1}: RF={r:.2%} | GRU={g:.2%}")

print(f"\n[RQ3] FEATURE IMPORTANCE & STABILITY")
print(f"Global Top Feature: {feat_imp.iloc[0]['Feature']} (Imp: {feat_imp.iloc[0]['Imp']:.4f})")
print(f"Behavioral Features in Top-5: {behavioral_in_top5}/5")
print(f"Temporal Stability (Quarterly Top-3 Features):")
for q_idx, top3 in enumerate(quarterly_top3):
    print(f"  Q{q_idx+1}: {', '.join(top3)}")
print(f"Stable Features (>=3 quarters): {stable_count}/{len(feature_frequency)}")

print(f"\n[RQ4] ECONOMIC PERFORMANCE (Risk-Adjusted)")
print(f"{'Metric':<15} | {'RF':<10} | {'GRU':<10} | {'Target':<10}")
print("-" * 50)
print(f"{'Sharpe Ratio':<15} | {rf_metrics['sharpe']:<10.3f} | {gru_metrics['sharpe']:<10.3f} | {CONFIG['target_sharpe']:<10}")
print(f"{'Total Return':<15} | {rf_metrics['total_return']*100:<10.2f}% | {gru_metrics['total_return']*100:<10.2f}% | -")
print(f"{'Win Rate':<15} | {rf_metrics['win_rate']*100:<10.1f}% | {gru_metrics['win_rate']*100:<10.1f}% | -")
print(f"{'Trades':<15} | {rf_metrics['n_trades']:<10} | {gru_metrics['n_trades']:<10} | -")

# 6. VISUALIZATION

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Bitcoin Price Prediction - Final Thesis Results', fontsize=16, fontweight='bold')

# Accuracy

ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(['RF', 'GRU'], [rf_da, gru_da], color=['#3498db', '#e74c3c'], edgecolor='black', alpha=0.8)
ax1.set_ylim(0.4, 0.6)
ax1.set_ylabel('Directional Accuracy', fontweight='bold')
ax1.set_xlabel('Model', fontweight='bold')
ax1.set_title(f'RQ1: Accuracy (p={p_val:.3f})', fontweight='bold')
ax1.axhline(0.5, ls='--', color='gray')
ax1.grid(axis='y', alpha=0.3)

# Stability

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(np.arange(4)-0.2, rf_q, 0.4, label='RF', color='#3498db', edgecolor='black', alpha=0.8)
ax2.bar(np.arange(4)+0.2, gru_q, 0.4, label='GRU', color='#e74c3c', edgecolor='black', alpha=0.8)
ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax2.set_ylabel('Accuracy', fontweight='bold')
ax2.set_xlabel('Quarter', fontweight='bold')
ax2.set_title('RQ2: Quarterly Stability', fontweight='bold')
ax2.legend() 
ax2.grid(axis='y', alpha=0.3)

# Feature Importance

ax3 = fig.add_subplot(gs[0, 2])
colors = ['#2ecc71' if 'tr' in f or 'trends' in f else '#3498db' for f in feat_imp['Feature']]
ax3.barh(feat_imp['Feature'], feat_imp['Imp'], color=colors, edgecolor='black', alpha=0.8)
ax3.invert_yaxis()
ax3.set_xlabel('Mean |SHAP|', fontweight='bold')
ax3.set_title('RQ3: Global Importance', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Sharpe

ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(['RF', 'GRU', 'Target'], [rf_metrics['sharpe'], gru_metrics['sharpe'], CONFIG['target_sharpe']],
        color=['#3498db', '#e74c3c', 'green'], edgecolor='black', alpha=0.8)
ax4.set_ylabel('Sharpe Ratio', fontweight='bold')
ax4.set_xlabel('Strategy', fontweight='bold')
ax4.set_title('RQ4: Sharpe Ratio', fontweight='bold')
ax4.axhline(0, color='black')
ax4.grid(axis='y', alpha=0.3)

# Cumulative Returns

ax5 = fig.add_subplot(gs[1, 1:])
ax5.plot(np.cumprod(1+gru_metrics['net_returns']), label='GRU Strategy', color='#e74c3c', lw=2.5)
ax5.plot(np.cumprod(1+rf_metrics['net_returns']), label='RF Strategy', color='#3498db', lw=2)
ax5.plot(np.cumprod(1+yte_aligned), label='Buy & Hold', color='gray', ls='--', lw=1.5)
ax5.set_xlabel('Trading Days', fontweight='bold')
ax5.set_ylabel('Cumulative Return', fontweight='bold')
ax5.set_title('Cumulative Returns (Net of Costs)', fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# Training

ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(history.history['loss'], label='Train')
ax6.plot(history.history['val_loss'], label='Val')
ax6.set_xlabel('Epoch', fontweight='bold')
ax6.set_ylabel('MSE Loss', fontweight='bold')
ax6.set_title('GRU Training Process', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# Scatter

ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(yte_aligned[:100], gru_pred_test[:100], alpha=0.6, color='#e74c3c', label='GRU', edgecolor='black', s=30)
ax7.scatter(yte_aligned[:100], rf_pred_aligned[:100], alpha=0.6, color='#3498db', label='RF', edgecolor='black', s=30)
lims = [min(yte_aligned[:100].min(), rf_pred_aligned[:100].min(), gru_pred_test[:100].min()),
        max(yte_aligned[:100].max(), rf_pred_aligned[:100].max(), gru_pred_test[:100].max())]
ax7.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='Perfect')
ax7.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
ax7.axvline(0, color='gray', linewidth=0.8, alpha=0.5)
ax7.set_xlabel('Actual Returns', fontweight='bold')
ax7.set_ylabel('Predicted Returns', fontweight='bold')
ax7.set_title('Prediction Alignment', fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# RANK Heatmap

ax8 = fig.add_subplot(gs[2, 2])
stability_matrix_ranks = np.zeros((len(feat_cols), 4))
for q_idx in range(4):
    s = q_idx * q_size
    e = (q_idx+1) * q_size if q_idx < 3 else len(Xte)
    shap_q = np.abs(shap_values[s:e]).mean(axis=0)
    top3_indices = np.argsort(shap_q)[-3:][::-1]
    for rank, feat_idx in enumerate(top3_indices):
        stability_matrix_ranks[feat_idx, q_idx] = 3 - rank
im = ax8.imshow(stability_matrix_ranks, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
ax8.set_xticks(range(4))
ax8.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax8.set_yticks(range(len(feat_cols)))
ax8.set_yticklabels(feat_cols, fontsize=8)
ax8.set_xlabel('Quarter', fontweight='bold')
ax8.set_title('RQ3: Feature Stability (Top-3 Rank)', fontweight='bold')
cbar = plt.colorbar(im, ax=ax8, ticks=[0, 1, 2, 3], label='Rank Score')
cbar.set_label('Rank Score', rotation=270, labelpad=15, fontsize=9, fontweight='bold')
ax8.grid(which='minor', color='white', linestyle='-', linewidth=2)

plt.tight_layout()
plt.savefig('figures/bitcoin_prediction_results_FINAL.png', dpi=300, bbox_inches='tight')
print("\n DONE. Visualization saved to figures/bitcoin_prediction_results_FINAL.png")

end_time = datetime.now()
runtime = (end_time - start_time).total_seconds()
print(f"Execution time: {runtime:.1f} seconds")
print(f"Completion: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("Author:Mohamed Mekky")