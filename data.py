import numpy as np
import tensorflow as tf

def get_data():
    """
    获取logistic数据.训练数据为1000份含有噪音的数据集。
    测试数据集为50份同样包含噪音的数据。
    :return:train_x：训练中输入样本x，shape为[1000,3]
            train_y：训练集中的输出样本y
            test_x：测试机中输入样本x
            test_y：测试样本中输出样本y
    """
    train_x1 = np.random.uniform(-200,200,1000).reshape([1000,1])
    train_x2 = np.random.uniform(-100,100,1000).reshape([1000,1])
    train_x3 = np.random.uniform(200,500,1000).reshape([1000,1])
    # 将上面的x1,x2,x3矩阵转化为1000*3的矩阵
    train_x = np.hstack((train_x1,train_x2,train_x3))
    train_noise = np.random.uniform(-1,1,1000)
    train_y = 0.23*train_x1**3 + 5*train_x2**7 - 79*train_x3 + 3 + train_noise

    test_x1 = np.random.uniform(-200, 200, 1000).reshape([1000, 1])
    test_x2 = np.random.uniform(-100, 100, 1000).reshape([1000, 1])
    test_x3 = np.random.uniform(200, 500, 1000).reshape([1000, 1])
    # 将上面的x1,x2,x3矩阵转化为1000*3的矩阵
    test_x = np.hstack((test_x1,test_x2,test_x3))
    test_noise = np.random.uniform(-1, 1, 50)
    test_y = 0.23*test_x1**3 + 5*test_x2**7 - 79*test_x3 + + 3 + test_noise

    return train_x,train_y,test_x,test_y

