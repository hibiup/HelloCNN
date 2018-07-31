from unittest import TestCase

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 添加一个 activation 神经层
def add_layer(inputs, in_size, out_size, activation_func=None, name='Layer', summarizer=None, keep_drop=None):
    """
    :param inputs:             输入的数据集
    :param in_size:            输入的神经点（对应输入数据集中每条记录的大小）
    :param out_size:           输出结果的尺寸
    :param activation_func:    激励函数
    :param name:
    :param summarizer:
    :return:
    """
    with tf.name_scope(name):
        with tf.name_scope('Weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]))   # 生成随机矩阵(层) shape = in_size x out_size
            if summarizer is not None: summarizer.histogram('Weights', weights)                       # Display in 'HISTOGRAM' and 'DISTRIBUTIONS'

        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)            # 数组  shape = 1 x out_size，推荐初始值不为0，因此 + 0.1
            if summarizer is not None: summarizer.histogram('Biases', biases)

        with tf.name_scope('Predict'):
            predict = tf.add(tf.matmul(inputs, weights), biases)           # 层运算（线形）

        if keep_drop is not None:
            predict = tf.nn.dropout(predict, keep_drop)

        outputs = predict if activation_func is None else activation_func(predict)  # 如果指定了 activation，则激励运算结果
        if summarizer is not None: summarizer.histogram('Outputs', outputs)
    return outputs


def init_summerizer(session, summarizer, output_paths):
    merged = summarizer.merge_all
    writers = {writer_id: summarizer.FileWriter(output_path, session.graph) for writer_id, output_path in output_paths.items()}

    def merge_summery(writer_id, index, feed_dict):
        nonlocal writers, merged
        summary = session.run(merged(), feed_dict=feed_dict)
        writers.get(writer_id).add_summary(summary, index)

    return merge_summery


def compute_accuracy(session, predict, test_feed_dict):
    y_verify = session.run(predict, feed_dict=test_feed_dict)            # 尝试预测一下
    correct_prediction = tf.equal(tf.argmax(y_verify, 1), tf.argmax(list(test_feed_dict.values())[1], 1))  # 将预测最大值与测试数据的最大值比较
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   # 得到百分比
    result = session.run(accuracy, feed_dict=test_feed_dict)
    return result


def visualization(x, y, predictions):
    from itertools import cycle
    col_gen = cycle('bgrcmk')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    plt.ion()
    plt.show()

    line = None
    for pred in predictions:
        line = ax.plot(x, pred, color=f'C{np.random.randint(1, 10)}')    # 加入每一轮的预测数据（线）
        plt.pause(0.2)


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

        def generate_test_data():
            x_data = np.linspace(-1., 1., 300, dtype=np.float32)[:, np.newaxis]  # 生成随机测试数据( feature=1, record=300)
            noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)   # 增加 noise 只是为了使数据看上去更真实
            y_data = np.square(x_data) - 0.5 + noise          # 模拟 y 数据(大致的 x 平方关系)
            return x_data, y_data

        summarizer = tf.summary

        # 定义运算过程
        with tf.name_scope('Input'):
            x_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name="x_input")
            y_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name="y_input")

        layer_1_output = add_layer(x_placeholder, x_placeholder.shape[1].value, 10,   # 输入节点数与原数据的 feature 相同，输出节点假设为10个
                                   activation_func=tf.nn.relu,                        # !! test activation function: relu !!
                                   name='Layer_1',
                                   summarizer=summarizer)
        layer_predict = add_layer(layer_1_output, 10, y_placeholder.shape[1].value,   # 上层的输出书下层的输入，size 也相同
                                  name='Layer_2',
                                  summarizer=summarizer)

        # 计算输出结果s和“真实”值的残差
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - layer_predict), reduction_indices=1))
            summarizer.scalar("Loss", loss)         # Display in 'SCALARS'

        with tf.name_scope('Train'):
            # 准备神经网络
            optimizer = tf.train.GradientDescentOptimizer(0.1)     # Learning rate(学习效率)取值在 0 到 1 之间
            train_step = optimizer.minimize(loss)                  # 以尽可能减小误差的方向进行训练

        # 运算
        X, y = generate_test_data()
        ph_dict = {x_placeholder: X, y_placeholder: y}
        init = tf.initialize_all_variables()
        predictions = []
        with tf.Session() as s:
            merge = init_summerizer(s, summarizer, {"train": "summary"})

            s.run(init)
            for i in range(1000):
                s.run(train_step, feed_dict=ph_dict)
                if i % 100 == 0:           # 每 100 步打印出结果看一下
                    merge("train", i, ph_dict)

                    print(s.run(loss, feed_dict=ph_dict))
                    predictions.append(s.run(layer_predict, feed_dict=ph_dict))

        visualization(X, y, predictions)

    def test_tensorflow_classification(self):
        """
        用 Tensorflow 自带的数字图像数据测试分类学习
        """
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets("MNIST_data",   # MNIST_data 数据包如果不存在会自动下载
                                          one_hot=True)   # one_hot 因为十个数字，因此每个在 y_label 中占一位.

        # Define placeholder for input
        x_ph = tf.placeholder(tf.float32, [None, 784])    # None：不规定数据量，784：每张图片的像数点（28 x 28）
        y_ph = tf.placeholder(tf.float32, [None, 10])     # 因为有十个可能的数字，因此 one_hot 后会得到长度为 10 的列表

        summarizer = tf.summary

        prediction = add_layer(x_ph, 784, 10,                   # 输入矩阵等于图像尺寸 784，输出 10 个分类
                               activation_func=tf.nn.softmax,   # softmax 可以用于 classification
                               summarizer=summarizer)

        # loss 函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
        # 是配合 softmax 计算 loss 的一种方法
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ph * tf.log(prediction),    # loss
                                                      reduction_indices=[1]))
        summarizer.scalar("Loss", cross_entropy)

        # 使用 gradient descent 来训练模型, 最小化 loss
        training = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # 初始化变量
        with tf.Session() as s:
            s.run(tf.initialize_all_variables())
            merge = init_summerizer(s, summarizer, {"train": "summary/train", "test": "summary/test"})
            #merged = tf.summary.merge_all()
            #writer = tf.summary.FileWriter('summary/train', s.graph)
            #test_writer = tf.summary.FileWriter('summary/test', s.graph)


            for i in range(1000):
                x_data, y_labels = mnist.train.next_batch(100)    # 每次训练取 100 个不同的样品（而不是基于全部，这样可以加快训练速度）
                feed_dict = {x_ph: x_data, y_ph: y_labels}
                s.run(training, feed_dict=feed_dict)
                if i % 50 == 0:
                    test_feed_dict = {x_ph: mnist.test.images, y_ph: mnist.test.labels}

                    merge("train", i, feed_dict=feed_dict)
                    merge("test", i, feed_dict=test_feed_dict)
                    #summary = s.run(merged, feed_dict=feed_dict)
                    #writer.add_summary(summary, i)
                    #test_summary = s.run(merged, feed_dict=test_feed_dict)
                    #test_writer.add_summary(test_summary, i)

                    print(compute_accuracy(s, prediction, test_feed_dict))  # 打印出预测准确率

    def test_dropout_fix_overfitting(self):
        from sklearn.datasets import load_digits    # digits 是手写数字图片数据集（Bunch），包含 8*8 像素的图像集和一个[0, 9]整数的标签
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelBinarizer

        summarizer = tf.summary

        digits = load_digits()    # 1797 个样本

        # 随机获取一张图片看一下. 每张图片都是 8x8 格式的
        # plt.imshow(digits.images[np.random.randint(1797)])
        # plt.show()

        x, y = digits.data, LabelBinarizer().fit_transform(digits.target)   # LabelBinarizer 类似 one_hot，将这1~10变成10个“BIT”，每个“BIT”表示一个值
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

        # 定义占位符
        x_ph, y_ph = tf.placeholder(tf.float32, [None, 8*8]), tf.placeholder(tf.float32, [None, 10])

        feed_dict = {x_ph: x_train, y_ph: y_train}
        test_feed_dict = {x_ph: x_test, y_ph: y_test}

        # 生成网络层
        middle_number = 100
        layer_1 = add_layer(x_ph, 8*8,                      # 输入层 64 个节点(每张图片)
                            middle_number,                  # 中间层个数故意放大到 100 个，以制造 overfitting 的效果。
                            activation_func=tf.nn.tanh,     # https://blog.csdn.net/brucewong0516/article/details/78834332
                            summarizer=summarizer,
                            keep_drop=0.5)                  # keep_drop=0.5 的意思是每次运算只随机取 50% 的数据
                                                            # 因为我们故意生成了 100 个节点的网络来放大了拟合，可以通过 dropout 来
                                                            # 随机忽略一些对结果造成偏执影响的节点
        layer_prediction = add_layer(layer_1, middle_number,
                                     10,   # 输出归类为 10 个节点(10个数字)
                                     activation_func=tf.nn.softmax,
                                     summarizer=summarizer,
                                     keep_drop=0.5)
        # Loss
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ph * tf.log(layer_prediction), reduction_indices=[1]))
        summarizer.scalar("Loss", cross_entropy)

        # 定义 Train 算法
        training = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

        # 开始训练
        with tf.Session() as s:
            s.run(tf.initialize_all_variables())
            merge = init_summerizer(s, summarizer, {"train": "summary/train", "test": "summary/test"})

            for i in range(1000):
                s.run(training, feed_dict=feed_dict)      # 训练的时候只使用 50% 的节点
                if i % 50 == 0:
                    merge("train", i, feed_dict=feed_dict)  # 输出给 tensorboard 仍然是 100% 的信息
                    merge("test", i, feed_dict=test_feed_dict)
