import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# 读取 Excel 数据
data = pd.read_excel("POI.xlsx")

future_lengths = []
# 创建空的 DataFrame，用于保存预测结果
output_data = pd.DataFrame(columns=["Input", "Predicted"])
# 逐行预测并保存结果
for index, row in data.iterrows():
    # 输入数据
    X = np.array([[2005], [2015]])  # 两个年份的输入特征
    Y = np.array([row["POI_22"], row["POI_15"]])  # 对应的道路长度

    # 创建并训练线性回归模型
    model = LinearRegression()
    model.fit(X, Y)

    # 打印回归模型的参数 k 和 b
    k = model.coef_[0]
    b = model.intercept_
    print("回归模型的参数 k:", k)
    print("回归模型的参数 b:", b)
    year = 2022
    future_length = k * year + b
    future_lengths.append(future_length)
    output_data = output_data.append({"Input": row["POI_22"], "Input2": row["POI_15"],
                                      "Predicted": future_length}, ignore_index=True)

# 将预测结果保存到 CSV 文件
# output_data.to_csv("POI-predicted_data-22.csv", index=False)

# 读取预测后的数据
predicted_data = pd.read_csv("POI-predicted_data-05.csv")

# 创建空的 DataFrame，用于保存所有数据的拟合结果
fit_data = pd.DataFrame(columns=["Year", "POI"])

# 遍历每行数据并绘制拟合曲线
for index, row in predicted_data.iterrows():
    # 获取每行的数据
    input_values = [row["Input"], row["Input2"]]
    predicted_value = row["Predicted"]

    # 绘制每条曲线的拟合结果
    plt.plot([2005, 2015], input_values, marker='o', label='Input Data')
    plt.scatter(2022, predicted_value, color='red', label='Predicted Value')

    # 使用线性回归模型参数绘制拟合曲线
    k = (input_values[1] - input_values[0]) / (2015 - 2005)  # 计算斜率
    b = input_values[0] - k * 2005  # 计算截距

    x_vals = np.linspace(2022, 2005, 100)  # 生成 x 值
    y_vals = k * x_vals + b  # 根据线性方程计算对应的 y 值
    plt.plot(x_vals, y_vals, color='green', label='Fitted Line')

    # 添加到 DataFrame 中
    fit_data = fit_data.append({"Year": x_vals, "POI": y_vals}, ignore_index=True)

    plt.xlabel('Year')
    plt.ylabel('POI')
    plt.title(f'POI Prediction for Row {index}')
    plt.legend()
    plt.show()
