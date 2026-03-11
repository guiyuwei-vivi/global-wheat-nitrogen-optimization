import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, SpatialDropout1D

# ========= 读取数据 =========
data_x = pd.read_csv('SAMPLE_X(pad).csv')
data_y = pd.read_csv('SAMPLE_Y.csv')

# 数据预处理
X = data_x.values
y = data_y.values.ravel()  # 将y转换为一维数组

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 归一化（仅对 X）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 重塑数据以适应 CNN 形状 (N, L, C)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped  = X_test_scaled.reshape(X_test_scaled.shape[0],  X_test_scaled.shape[1],  1)

# ========= 构建带 Dropout 的 CNN 模型 =========
def build_model(input_len, dropout_p=0.3):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(input_len, 1)))
    model.add(MaxPooling1D(2))
    # 对通道做空间Dropout，更稳健
    model.add(SpatialDropout1D(dropout_p))

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_p))  # 关键：推理时也会被启用（MC Dropout）
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model(X_train_reshaped.shape[1], dropout_p=0.3)

# ========= 训练 =========
history = model.fit(
    X_train_reshaped, y_train,
    epochs=50, batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)

# ========= 常规评估 =========
loss = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"测试集上的损失: {loss}")
model.summary()

y_pred = model.predict(X_test_reshaped, verbose=0)
mse  = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"均方误差（MSE）: {mse}")
print(f"均方根误差（RMSE）: {rmse}")
print(f"平均绝对误差（MAE）: {mae}")
print(f"R平方（R²）: {r2}")

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='verification loss')
plt.title('Changes in loss during training')
plt.ylabel('loss'); plt.xlabel('period'); plt.legend(); plt.show()

# ========= MC Dropout 不确定性函数 =========
@tf.function
def _forward_with_dropout(m, xb, training=True):
    """强制在前向中启用 Dropout（training=True）"""
    return m(xb, training=training)

def mc_dropout_predict(model, X_reshaped, T=100, batch_size=4096, non_negative=True):
    """
    对每个样本进行 T 次随机前向传播（MC Dropout）
    返回:
      mean, std, ci95_lo, ci95_hi, all_samples   其中 all_samples 形状为 [N, T]
    若 non_negative=True，则对每次样本级预测先截断到 >=0 再统计。
    """
    N = X_reshaped.shape[0]
    preds = []

    for t in range(T):
        y_t = []
        for i in range(0, N, batch_size):
            xb = X_reshaped[i:i+batch_size]
            yb = _forward_with_dropout(model, xb, training=True)  # 启用dropout
            y_t.append(yb.numpy().reshape(-1))
        y_t = np.concatenate(y_t, axis=0)
        if non_negative:
            y_t = np.maximum(y_t, 0.0)
        preds.append(y_t)

    preds = np.stack(preds, axis=1)   # [N, T]
    mean = preds.mean(axis=1)
    std  = preds.std(axis=1, ddof=1)
    ci_lo = mean - 1.96 * std
    ci_hi = mean + 1.96 * std
    return mean, std, ci_lo, ci_hi, preds

# ========= 在新气候数据集上做“带不确定性”的预测 =========
# 加载测试（预测）数据集
precet_data = pd.read_csv('10model_n170%_585_2021-2040-pad.csv').values

# 预处理（与训练一致）
precet_scaled   = scaler.transform(precet_data)
precet_reshaped = precet_scaled.reshape(precet_scaled.shape[0], precet_scaled.shape[1], 1)

# MC Dropout 推理（T 次采样）
mean_pred, std_pred, lo95, hi95, samples = mc_dropout_predict(
    model, precet_reshaped, T=100, batch_size=4096, non_negative=True
)

# 同时也给出“单次常规预测”的版本（不带不确定性）
predictions = model.predict(precet_reshaped, verbose=0)
predictions[predictions < 0] = 0

# 保存结果
predictions_df = pd.DataFrame({
    'yw_mean': mean_pred,
    'yw_std':  std_pred,
    'ci95_lo': lo95,
    'ci95_hi': hi95
})
print(predictions_df.head())
predictions_df.to_csv('10model_n110%_126_2021-2040-PRE_with_uncertainty.csv', index=False)

# 对不确定性做一个直方图查看分布
plt.figure()
plt.hist(std_pred, bins=30)
plt.xlabel('Predictive std'); plt.ylabel('Count')
plt.title('Uncertainty (MC Dropout) for 10model_n110%_126_2021-2040')
plt.show()
