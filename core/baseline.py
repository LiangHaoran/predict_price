import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
import argparse
import xgboost as xgb
import logging
from sklearn import preprocessing, linear_model
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)

config_path = "/Users/haoran/Documents/BUAA_course/matchine_learnign/final_work/personal_work/predict_price/" \
              "config/config.json"
config = json.load(open(config_path))
raw_data = pd.read_csv(config['raw_data_path'])


"""定义日志等级，格式"""
logging.basicConfig(
                    level=logging.DEBUG,
                    filename=config['log_path'],
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


"""set parameters"""
ap = argparse.ArgumentParser()
ap.add_argument("--m", required=True, type=str)
args = vars(ap.parse_args())


def delet_nan(raw_data):
    """
    将数据中空值太多的特征去掉
    :param raw_data:
    :return:
    """
    data = raw_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', "Electrical"], axis=1, inplace=False)
    return data


def find_anomaly(column, threshold):
    """
    从column中找出大于threshold的元素，并返回索引
    """
    index = np.where(column > threshold)
    return pd.DataFrame(np.array(index))


def MAPE(true, pred):
    """
    计算相对百分比误差
    :param data:
    :return:
    """
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)


def convert_onehot(data, name):
    """
    将非数值特征做one hot编码
    :param data:原始数据矩阵
    :return:
    """
    out = LabelBinarizer().fit_transform(data[name])
    return np.array(out)


def conver_onehot_test(train, test, name):
    lab = LabelBinarizer()
    train_out = lab.fit_transform(train[name])
    test_out = lab.transform(test[name])
    return np.array(test_out)


def is_nan(column):
    """
    判断一列特征是否存在缺失值
    :param column:
    :return:存在缺失，返回True，不存在缺失，返回False
    """
    column = np.array(column)
    column = pd.DataFrame(column)
    print("是否存在缺失值(True-存在；False-不存在)")
    print(column.isnull().any())


def fill_nan_scalar(column):
    """
    用前一个和后一个数的均值来填补NAN
    :param column:
    :return:
    """
    index = np.where(np.isnan(column))
    index = np.array(index).reshape(-1)
    for i in range(index.shape[0]):
        column[index[i]] = np.nanmean([column[index[i]-1], column[index[i]+1]])


def fill_nan_label(column):
    """
    将缺失的部分看做新的类别，用AA填充
    :param column:
    :return:
    """
    column = column.fillna(value="AA")
    return column


def build_model(Model):
    if Model == "GBDT":
        model = GradientBoostingRegressor()
    if Model == "xgboost":
        model = xgb.XGBRegressor()
    if Model == "randomforest":
        model = RandomForestRegressor()
    return model, Model


"""将Alley，FireplaceQu，PoolQC，Fence，MiscFeature五个特征去掉"""
raw_data = delet_nan(raw_data)


"""将有缺失值的三个数值类字段fill"""
fill_nan_scalar(raw_data["LotFrontage"])
fill_nan_scalar(raw_data["MasVnrArea"])
fill_nan_scalar(raw_data["GarageYrBlt"])


"""类别特征的缺失值用“AA”填充，作为新的类别"""
raw_data["MasVnrType"] = fill_nan_label(raw_data["MasVnrType"])
raw_data["BsmtQual"] = fill_nan_label(raw_data["BsmtQual"])
raw_data["BsmtCond"] = fill_nan_label(raw_data["BsmtCond"])
raw_data["BsmtExposure"] = fill_nan_label(raw_data["BsmtExposure"])
raw_data["BsmtFinType1"] = fill_nan_label(raw_data["BsmtFinType1"])
raw_data["BsmtFinType2"] = fill_nan_label(raw_data["BsmtFinType2"])
raw_data["GarageType"] = fill_nan_label(raw_data["GarageType"])
raw_data["GarageFinish"] = fill_nan_label(raw_data["GarageFinish"])
raw_data["GarageQual"] = fill_nan_label(raw_data["GarageQual"])
raw_data["GarageCond"] = fill_nan_label(raw_data["GarageCond"])


"""所有数值特征和房价的散点图"""
scalar_name = ["MSSubClass", "LotFrontage", "LotArea", "OverallQual", "TotalBsmtSF", "BsmtFinSF2",
               "OverallCond", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
               "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
               "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
               "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
               "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]


"""数值特征对房价的重要度分析 - 随机森林"""
forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(raw_data[scalar_name], raw_data["SalePrice"])
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print("feature importances ----------------------- ")
for f in range(raw_data[scalar_name].shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, scalar_name[indices[f]], importances[indices[f]]))


"""将类别特征one hot编码"""
label_name = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
              "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle",
              "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond",
              "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
              "Heating", "HeatingQC", "CentralAir", "KitchenQual", "Functional",
              "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType",
              "SaleCondition"]
label_column = pd.DataFrame(convert_onehot(raw_data, label_name[0]))
for i in label_name:
    if i != "MSZoning":
        temp = convert_onehot(raw_data, i)
        label_column = pd.concat([pd.DataFrame(label_column), pd.DataFrame(temp)], axis=1)
label_column = np.array(label_column)


"""将处理好的数值特征和类别特征合并在一起，增加一步：将数值特征做归一化"""
min_max_scalar1 = preprocessing.MinMaxScaler()
final_data = pd.concat([pd.DataFrame(raw_data[scalar_name]), pd.DataFrame(label_column)], axis=1)
final_data = np.array(final_data)
print(final_data.shape)


"""划分数据集"""
target = raw_data["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(final_data, target, test_size=0.1, shuffle=True)


"""搭建baseline"""
"""模型：随机森林"""
target = raw_data["SalePrice"]
model, model_name = build_model(args["m"])
model.fit(x_train, y_train)
pre = model.predict(x_test)


"""计算指标"""
mse = metrics.mean_squared_error(np.array(y_test), np.array(pre))
rmse = np.sqrt(mse)
mape = MAPE(np.array(y_test), np.array(pre))
r2 = r2_score(np.array(pre), np.array(y_test))

"""验证集结果可视化"""
plt.title("mse: %s    rmse: %s    mape: %s    r2: %s" % (mse, rmse, mape, r2))
plt.plot(np.array(pre), label="pre")
plt.plot(np.array(y_test), label="target")
plt.legend()
plt.savefig('/Users/haoran/Documents/BUAA_course/matchine_learnign/final_work/personal_work/predict_price/figure/' +
            args['m'] + '_prediction.png')
plt.show()

plt.title("local")
plt.plot(np.array(pre)[50:70], label="pre")
plt.plot(np.array(y_test)[50:70], label="target")
plt.legend()
plt.savefig('/Users/haoran/Documents/BUAA_course/matchine_learnign/final_work/personal_work/predict_price/figure/' +
            args['m'] + '_prediction.png')
plt.show()

logging.debug('model={}'.format(model_name))
logging.debug('mse={}'.format(mse))
logging.debug('rmse={}'.format(rmse))
logging.debug('mape={}'.format(mape))
logging.debug('r2={}'.format(r2))


"""测试"""
test_data = pd.read_csv(config['test_data_path'])


"""去掉训练集中缺失严重的5个特征，然后看一下数据情况"""
test_data = delet_nan(test_data)


"""填补数值变量NAN"""
fill_nan_scalar(test_data["LotFrontage"])
fill_nan_scalar(test_data["MasVnrArea"])
fill_nan_scalar(test_data["GarageYrBlt"])


"""类别特征的缺失值用“AA”填充，作为新的类别"""
test_data["MasVnrType"] = fill_nan_label(test_data["MasVnrType"])
test_data["BsmtQual"] = fill_nan_label(test_data["BsmtQual"])
test_data["BsmtCond"] = fill_nan_label(test_data["BsmtCond"])
test_data["BsmtExposure"] = fill_nan_label(test_data["BsmtExposure"])
test_data["BsmtFinType1"] = fill_nan_label(test_data["BsmtFinType1"])
test_data["BsmtFinType2"] = fill_nan_label(test_data["BsmtFinType2"])
test_data["GarageType"] = fill_nan_label(test_data["GarageType"])
test_data["GarageFinish"] = fill_nan_label(test_data["GarageFinish"])
test_data["GarageQual"] = fill_nan_label(test_data["GarageQual"])
test_data["GarageCond"] = fill_nan_label(test_data["GarageCond"])


"""将类别特征one hot编码"""
label_column_test = pd.DataFrame(conver_onehot_test(raw_data, test_data, label_name[0]))
for i in label_name:
    if i != "MSZoning":
        temp1 = conver_onehot_test(raw_data, test_data, i)
        label_column_test = pd.concat([pd.DataFrame(label_column_test), pd.DataFrame(temp1)], axis=1)
label_column_test = np.array(label_column_test)


"""将数值特征和类别特征合并在一起，增加对数值特征进行归一化"""
final_test_data = pd.concat([pd.DataFrame(test_data[scalar_name]), pd.DataFrame(label_column_test)], axis=1)
final_test_data = np.array(final_test_data)


"""用训练好的模型预测"""
prediction = model.predict(final_test_data)
prediction = np.array(prediction)
plt.plot(prediction, label="prediction of test data")
plt.legend()
plt.savefig('/Users/haoran/Documents/BUAA_course/matchine_learnign/final_work/personal_work/predict_price/figure/' +
            args['m'] + '_result.png')
plt.show()


"""将测试集结果保存到本地"""
id = test_data["Id"]
out = pd.concat([pd.DataFrame(id), pd.DataFrame(prediction, columns=["SalePrice"])], axis=1)

wd = pd.DataFrame(out)
wd.to_csv(config["result_save_path"], index=None)
print("saved to:", config["result_save_path"])

