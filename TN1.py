import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.regularizers import l1, l2
import os, csv

# 数据读取
data = pd.read_csv("C:/Documents/AHU/AI/ANN_demo/data_demo.CSV")
# print(data.head())

# 数据预处理
data = pd.get_dummies(data, columns=["C"])  # 对C种类独热码处理
data = data.drop(["NO3N", "NO2N"], axis=1)  # 处理成唯一标签
print(data.head())

data = np.array(data)  # 数据格式转化为numpy数组
# print(data)
# print(data.shape)

# 归一化
data_label = data[:, 2]
data_feature = np.delete(data, 2, axis=1)
print(f"data_feature: {data_feature}")
print(f"data_label: {data_label}")

scaler_1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_feature_scale = scaler_1.fit_transform(data_feature)

scaler_2 = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_label_scale = scaler_2.fit_transform(data_label.reshape(-1, 1))


# 6:2:2 划分训练集、交叉验证集、测试集
train_features, test_features, train_labels, test_labels = train_test_split(
    data_feature_scale, data_label_scale, test_size=0.15
)
train_features, val_features, train_labels, val_labels = train_test_split(
    train_features, train_labels, test_size= 3/17
)

# print(len(train_features))
# print(len(test_features))
# print(len(val_features))
# print(len(train_labels))
# print(len(test_labels))
# print(len(val_labels))


# 神经网络构建
class DNBF(Model):
    def __init__(self):
        super(DNBF, self).__init__()
        self.d1 = Dense(3, activation="relu", kernel_regularizer=l2(0.005))
        self.d2 = Dense(1, activation="linear")

    def call(self, x):
        x = self.d1(x)
        y = self.d2(x)
        return y


model = DNBF()

# 创建指数衰减学习率
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.96
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=exponential_decay),
    loss =tf.keras.losses.MSE
)

batch_size = 32
epochs = 1000

# 终端 tensorboard --logdir mylogs
log_dir = "C:/Documents/AHU/AI/ANN_demo/neurals_num/tn1_adam"
callback = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10),
]

model.fit(
    train_features,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_features, val_labels),
    validation_freq=1,
    callbacks=[callback],
)

model.summary()

# 模型验证
predict = model.predict(test_features)
# print(predict.shape)
# print(test_labels.shape)
# print(f"predict: {predict}")
# print(f"test_labels: {test_labels}")

R2_1 = r2_score(test_labels, predict)
MSE_1 = mean_squared_error(test_labels, predict)
MAE_1 = mean_absolute_error(test_labels, predict)
print(f"R2_1: {R2_1}")
print(f"mse_1: {MSE_1}")
print(f"mae_1: {MAE_1}")

# 预测结果的反归一化
test_labels_inver = scaler_2.inverse_transform(test_labels)
predict_inver = scaler_2.inverse_transform(predict.reshape(-1, 1))

# print(f"test_labels_inver: {test_labels_inver}")
# print(f"predict_inver: {predict_inver}")

R2_2 = r2_score(test_labels_inver, predict_inver)
MSE_2 = mean_squared_error(test_labels_inver, predict_inver)
MAE_2 = mean_absolute_error(test_labels_inver, predict_inver)
print(f"R2_2: {R2_2}")
print(f"MSE_2: {MSE_2}")
print(f"MAE_2: {MAE_2}")

#模型保存
model.save(
    "C:/Documents/AHU/AI/ANN_demo/neurals_num/modelsave_tn1_adam/", save_format="tf"
)
