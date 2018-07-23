from unittest import TestCase

import tensorflow as tf
import numpy as np


class TestTensorflow(TestCase):
    """
    Tensorflow 例子
    """

    def test_variable(self):
        """
        Tensorflow 的变量和常量运算
        """
        state = tf.Variable(100, name="counter")                  # 定义变量
        print(id(state), state.name, state.value())                          # output: counter:0 Tensor("counter/read:0", shape=(), dtype=int32)
        one = tf.constant(1)                                      # 常量
        print(one)

        new_value = tf.add(state, one)                            # 加法运算
        update = tf.assign(state, new_value)                      # 赋值运算，将 new_value 赋予 state， 相当于 state = state + one
                                                                  # Tensorflow 中的运算都是 lazy 的，因此此处并没有真的执行，
                                                                  # 直到下面被赋予 session.run 之后.

        # 如果过程中存在变量，就必须执行以下命令
        init = tf.initialize_all_variables()

        # 运算。 Tensorflow 的 run() 函数就如同 shell，大部分指令都通过 run() 函数来执行。
        with tf.Session() as session:
            session.run(init)                                     # 执行变量初始化
            for _ in range(3):
                session.run(update)                               # 执行上面定义的 assign 运算（执行两次）
                print(id(state), session.run(state))              # Tensorflow 的取值运算也是 run()，相当于在 shell 打印出值
                                                                  # 同时也可以看到 state 的 id 没有变化，说明 Variable 是 mutable

    def test_run_with_matrix_multiply(self):
        """
        矩阵运算: matrix_left x marix_right
        """
        # 定义两个常量
        matrix_left  = tf.constant([[3, 3]])                      # Matrix axis = x
        matrix_right = tf.constant([[2], [2]])                    # Matrix axis = y

        product = tf.matmul(matrix_left, matrix_right)            # matrix multiply = np.dot(left, right)

        # 执行
        with tf.Session() as session:
            result = session.run(product)
            print(result)

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
