import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer




config_path = "/Users/haoran/Documents/BUAA_course/matchine_learnign/final_work/personal_work/predict_price/" \
              "config/config.json"
config = json.load(open(config_path))
raw_data = pd.read_csv(config['raw_data_path'])

print(raw_data.columns.shape)


def delet_nan(raw_data):
    """
    将数据中空值太多的特征去掉
    :param raw_data:
    :return:
    """
    data = raw_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
    return data







def convert_onehot(data):
    """
    将非数值特征做one hot编码
    :param data:原始数据矩阵
    :return:
    """
    out = LabelBinarizer().fit_transform(data["Neighborhood"])
    return out



out = convert_onehot(raw_data)
print(out.shape)
print(out[4, :])

data = delet_nan(raw_data)
print(data.columns.shape)

