from unittest import TestCase

import tensorflow as tf
import numpy as np


class TestTensorflow(TestCase):
    def test_one_dimensional_equation(self):
        """
        用 Tensorflow 来预测一元一次函数（y=x*a + b）的权重和偏差值
        """
        weight = 0.1
        bias = 0.3

        # 1) Create test function
        x_data = np.random.rand(100).astype(np.float32)
        y_data = x_data * weight + bias                           # 得到 100 个用于训练的结果

        # 2) 用 Tensorflow 来构建以上 function
        Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 定义一个 Tensorflow Variable 的权重值变量(一维)。范围随机，
                                                                  # 这个变量就是训练的目标变量，会在训练过程中被逐步修正。
        Biases = tf.Variable(tf.zeros([1]))                       # 定义一个 Tensorflow Variable 的 bias 值变量(一维)。初始化为0
        y = Weights * x_data + Biases                             # 用 Tensorflow Variable 来构建公式，返回用于评测的 Tensor(也是 100 个)

        loss = tf.reduce_mean(tf.square(y - y_data))              # 用方差来为每次预测计算损失，返回每次的 loss
        optimizer = tf.train.GradientDescentOptimizer(0.5)        # 定义一个优化器，参数是一个学习效率值，值区间: (1, 0)
        train = optimizer.minimize(loss)                          # 使用优化器来生成训练模型

        # 3) 初始化神经网络
        init = tf.initialize_all_variables()                      # 初始化神经网络(为神经网络的每个节点生成随机值)
        session = tf.Session()
        session.run(init)                                         # 启动神经网络

        # 4) s开始训练
        for step in range(201):                                   # 训练若干次
            session.run(train)
            if step % 20 == 0:                                    # 每20次输出一次当前 wight, bias 值
                print(step, session.run(Weights), session.run(Biases))      # session.run(Weights) 返回当前值

        # 停止
        session.close()

    def test_matrix_multiply(self):
        """
        matrix_left x marix_right
        """
        # 定义
        matrix_left  = tf.constant([[3, 3]])                      # Matrix axis = x
        matrix_right = tf.constant([[2], [2]])                    # Matrix axis = y

        product = tf.matmul(matrix_left, matrix_right)            # matrix multiply = np.dot(left, right)

        # 执行
        session = tf.Session()
        result = session.run(product)

        print(result)

        # 停止
        session.close()
