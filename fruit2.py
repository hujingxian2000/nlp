# -*- coding = utf-8 -*-
# @Time :2022/12/15 0:38
# @Author :Hu Jingxian
# @File :fruit2.py
# @Software :PyCharm

import os
from skimage import io,transform
import numpy as np
from sklearn.utils import shuffle
import keras
import tensorflow as tf

#如果对所有种类进行训练和测试，选择全部Training和Test
dir_path1 = 'fruits-360/Training1'
dir_path2 = 'fruits-360/Test1'
#读取数据
def load_data(dir_path):
    images = []
    labels = []
    file = os.listdir(dir_path)     #返回一个列表：包含dir_path里的文件名，即分类
    n_classes = len(file)           #分类的数量
    n = 0
    for l in file:
        img = os.listdir(dir_path+'/'+ l)
        for i in img:
            img_path = dir_path+'/'+ l+'/'+ i
            labels.append(int(n))
            images.append(io.imread(img_path))
        n += 1
    return images,labels,n_classes

images,labels,n_classes = load_data(dir_path1)
images_test,labels_test,_ = load_data(dir_path2)
print("种类：",n_classes)

#数据预处理：
def data_processing(images,labels,n_classes):
    train_x = np.array(images)
    train_y = np.array(labels)
    indx = np.arange(0,train_y.shape[0])
    indx = shuffle(indx)        #随机打乱
    train_x = train_x[indx]
    train_y = train_y[indx]
    train_y = keras.utils.to_categorical(train_y,n_classes)     #转为Onehot标签
    return train_x,train_y

train_x,train_y = data_processing(images,labels,n_classes)
test_x,test_y = data_processing(images_test,labels_test,n_classes)

#神经网络参数配置
batch_size = 32
dropout = 0.8
training_epochs = 5
learningrate = 0.001

filter_height = 5
filter_width = 5

in_channels = 3
out_channels1 = 32
out_channels2 = 64

image_size = train_x.shape[1] #图片尺寸
total_data_num = train_x.shape[0]

stride1 = 2
stride2 = 2

x = tf.placeholder(tf.float32,[None,image_size,image_size,in_channels])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

Weights = {"conv1":tf.Variable(tf.random_normal([filter_height,filter_width,in_channels,out_channels1])),
           "conv2":tf.Variable(tf.random_normal([filter_height,filter_width,out_channels1,out_channels2])),
           "conv3":tf.Variable(tf.random_normal([filter_height,filter_width,out_channels2,n_classes])),}

bias = {"conv1":tf.Variable(tf.random_normal([out_channels1])),
        "conv2":tf.Variable(tf.random_normal([out_channels2])),
        "conv3":tf.Variable(tf.random_normal([n_classes]))}


#定义卷积层的生成函数
def conv2d(x, W, b, stride=1):
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

#定义池化层的生成函数
def maxpool2d(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding="SAME")

#定义卷积神经网络生成函数
def conv_net(x, Weights, bias, dropout):
    #卷积层1
    conv1 = conv2d(x, Weights['conv1'], bias['conv1'])   ##100*100*32
    conv1 = maxpool2d(conv1, stride1)   ##经过池化层1 shape：50*50*32

    #卷积层2
    conv2 = conv2d(conv1, Weights['conv2'], bias['conv2'])  ##50*50*64
    conv2 = maxpool2d(conv2, stride2)  ##经过池化层2 shape:25*25*64

    #Dropout层防止预测数据过拟合
    conv2 = tf.nn.dropout(conv2, dropout)

    #全局平均池化层
    conv3 = conv2d(conv2, Weights['conv3'], bias['conv3'])  ##25*25*10
    prediction = tf.nn.avg_pool(conv3,ksize=[1,25,25,1],strides=[1,25,25,1],padding='SAME')

    return tf.reshape(prediction,[-1,n_classes])

#优化预测准确率
prediction = conv_net(x, Weights, bias, keep_prob)  #生成卷积神经网络
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))  #交叉熵损失函数
optimizer = tf.train.AdamOptimizer(learningrate).minimize(loss)  #优化器

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#初始会话并开始训练过程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #启动循环开始训练
    for epoch in range(training_epochs):
        total_batch = int(total_data_num / batch_size)
        #遍历全部数据集
        for i in range(total_batch):
            batch_x, batch_y = train_x[i*batch_size:batch_size * (i+1),:],train_y[i*batch_size:batch_size * (i+1),:]
            _, acc,Loss = sess.run([optimizer,accuracy,loss], feed_dict={x: batch_x,y: batch_y,keep_prob:dropout})
            # Compute average loss
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(Loss), "Training accuracy", "{:.5f}".format(acc))
    print(" Finished!")
    #测试模型
    test_feed = {x: test_x[0:100],y:test_y[0:100],keep_prob:1}
    print('Testing Accuracy:', sess.run(accuracy, feed_dict=test_feed))

