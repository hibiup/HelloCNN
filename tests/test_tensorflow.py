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
        print(id(state), state.name, state.value())               # output: counter:0 Tensor("counter/read:0", shape=(), dtype=int32)
        one = tf.constant(1)                                      # 常量
        print(one)

        new_value = tf.add(state, one)                            # 加法运算
        update = tf.assign(state, new_value)                      # 赋值运算，将 new_value 赋予 state， 相当于 state = state + one
                                                                  # Tensorflow 中的运算都是 lazy 的，因此此处并没有真的执行，
                                                                  # 直到下面被赋予 session.run 之后.

        # !! 如果过程中存在变量，就必须执行以下命令 !!
        init = tf.initialize_all_variables()

        """
         运算。 Tensorflow 的 run() 函数就如同 shell，大部分指令都通过 run() 函数来执行。
         因此以下过程相当于在 Tensorflow 中执行:
         > init     -- 初始化变量空间
         > update   -- 执行运算
         > state    -- 察看结果
        """
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
        matrix_left = tf.constant([[3, 3]])                      # Matrix axis = x
        matrix_right = tf.constant([[2], [2]])                    # Matrix axis = y

        product = tf.matmul(matrix_left, matrix_right)            # matrix multiply = np.dot(left, right)

        # 执行
        with tf.Session() as session:
            result = session.run(product)
            print(result)

    def test_placeholder(self):
        """
        测试 place holder
        """
        input1 = tf.placeholder(tf.float32)                       # 定义 place holder，给定类型
        input2 = tf.placeholder(tf.float32)
        output = tf.multiply(input1, input2)                      # 输出为乘法运算

        with tf.Session() as s:
            res = s.run(output, feed_dict={input1: [7.], input2: [2.]})  # place holder 在作为 run 的参数，以 dict 的形式传入
            print(res)                                            # [14.]

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
        weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 定义一个 Tensorflow Variable 的权重值变量(shape=1)。范围随机，
                                                                  # 这个变量就是训练的目标变量，会在训练过程中被逐步修正。
        biases = tf.Variable(tf.zeros([1]))                       # 定义一个 Tensorflow Variable 的 bias 值变量(shape=1)。初始化为0
        y = weights * x_data + biases                             # 用 Tensorflow Variable 来构建公式，返回用于评测的 Tensor(也是 100 个)

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
                print(step, session.run(weights), session.run(biases))      # session.run(Weights) 返回当前值

        # 停止
        session.close()

    def test_activation_function(self):
        """
        激励函数: https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-6-A-activation-function/

        y = N * x 是一个线形函数(linear) =〉 y = AF(N * x) 中 AF 是激励函数，将等式变成非线性的. 激励函数的作用是用来放大或缩小
        特征，常见的激励模式包括 linear, step, ramp 等. 激励函数必须是可积的，一般从神经网络的第二层以后开开始。

        参考：
        https://www.tensorflow.org/api_guides/python/nn
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation
        """

        # 添加一个 activation 神经层
        def add_layer(inputs, in_size, out_size, activation_func=None):
            weights = tf.Variable(tf.random_normal([in_size, out_size]))   # 生成随机矩阵(层) shape = in_size x out_size
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)            # 数组  shape = 1 x out_size，推荐初始值不为0，因此 + 0.1

            predict = tf.matmul(inputs, weights) + biases                 # 层运算（线形）
            outputs = predict if activation_func is None else activation_func(predict)  # 如果指定了 activation，则激励运算结果

            return outputs

        def generate_test_data():
            x_data = np.linspace(-1., 1., 300, dtype=np.float32)[:, np.newaxis]  # 生成随机测试数据( feature=1, record=300)
            noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)   # 增加 noise 只是为了使数据看上去更真实
            y_data = np.square(x_data) - 0.5 + noise          # 模拟 y 数据(大致的 x 平方关系)
            return x_data, y_data

        # 定义运算过程
        x_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
        y_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

        layer_1_output = add_layer(x_placeholder, x_placeholder.shape[1].value, 10,  # 输入节点数与原数据的 feature 相同，输出节点假设为10个
                                   activation_func=tf.nn.relu)
        y_pred = add_layer(layer_1_output, 10, y_placeholder.shape[1].value)         # 上层的输出书下层的输入，size 也相同

        # 计算输出结果s和“真实”值的残差
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - y_pred), reduction_indices=1))

        # 准备神经网络
        optimizer = tf.train.GradientDescentOptimizer(0.1)     # Learning rate(学习效率)取值在 0 到 1 之间
        train_step = optimizer.minimize(loss)                  # 以尽可能减小误差的方向进行训练

        # 运算
        X, y = generate_test_data()
        ph_dict = {x_placeholder: X, y_placeholder: y}
        init = tf.initialize_all_variables()
        with tf.Session() as s:
            s.run(init)
            for i in range(1000):
                s.run(train_step, feed_dict=ph_dict)
                if i % 100 == 0:           # 每 100 步打印出结果看一下
                    print(s.run(loss, feed_dict=ph_dict))
