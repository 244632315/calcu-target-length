#输入大板内成品小板总块数i_length_slot，轧制目标长度roll_length，
#头部试材长度 hcrop	尾部试材长度tcrop，切缝量d_cut_width_i
#输出切头尾量len_crop
#批量注释ctrl+/

print ('hello world!')
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#建立隐藏层
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size],dtype=tf.float32))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1,dtype=tf.float32)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases#行数是样例个数，列数是特征个数
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

#读取数据
data=pd.read_csv("data_train.csv")
data_test_pd=pd.read_csv("data_test.csv")

data_train=np.array(data[['roll_thick','roll_width','corect_length' ]])#,'tcrop','hcrop'
data_train=data_train.astype(np.float32)
print(data_train.shape,data_train .dtype)#输入训练特征数据
y_data=np.array(data[['len_crop']])
y_data=y_data.astype(np.float32)
#print(y_data,y_data.shape)#输入正确结果
#print(data_train,data_train.shape)

data_test=np.array(data_test_pd[['roll_thick','roll_width','corect_length']])#,'tcrop','hcrop'
data_test=data_test.astype(np.float32)#输入测试数据集
y_test=np.array(data_test_pd[['len_crop']])
y_test=y_test.astype(np.float32)

l0_num=3
##create TensorFlow structure start
xs=tf.placeholder(tf.float32,[None,l0_num])
ys=tf.placeholder(tf.float32,[None,1])


#l1_num=35
l1_num=35
l2_num=35
l3_num=35

l1=add_layer(xs,l0_num,l1_num,activation_function=tf.nn.tanh)#tf.nn.relu
#l2=add_layer(l1,l1_num,l2_num,activation_function=tf.nn.tanh)#tf.nn.relu
#l3=add_layer(l2,l2_num,l3_num,activation_function=tf.nn.tanh)#tf.nn.relu
# l4=add_layer(l3,10,10,activation_function=tf.nn.tanh)#tf.nn.relu



prediction=add_layer(l1,l1_num,1,activation_function=None)

# l2=add_layer(l1,l1_num,l2_num,activation_function=tf.nn.tanh)#tf.nn.relu
# prediction=add_layer(l2,l2_num,1,activation_function=None)


print(np.shape(prediction),np.shape(ys))


loss=tf.reduce_mean(tf.square(prediction-ys),name="loss")#,reduction_indices=[1]
optimizer=tf.train.GradientDescentOptimizer(0.001)
train=optimizer.minimize(loss,name="train")

##初始化
init=tf.initialize_all_variables()
##create TensorFlow structure end


#执行初始化
sess=tf.Session()
sess.run(init)##very important

writer=tf.summary.FileWriter('./graphs',sess.graph)
train_times=20000

#结果可视化
fig=plt.figure()#生成图片框
ax=fig.add_subplot(1,1,1)
ax.scatter(np.array(data[['roll_width' ]]),y_data)
#ax=Axes3D(fig)
#ax.scatter3D(data[['roll_width' ]],data[['corect_length' ]],y_data)
plt.ion()
plt.show()
#结果可视化-end

for step in range(train_times):
    sess.run(train,feed_dict={xs:data_train,ys:y_data})
    if step==1 :
        print(step, sess.run(xs, feed_dict={xs: data_train, ys: y_data}))
        print(step, sess.run(ys, feed_dict={xs: data_train, ys: y_data}))
    if step % 100 == 0:

        print(step,sess.run(loss,feed_dict={xs:data_test,ys:y_test}))#输出误差
        #输出结果可视化
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction, feed_dict={xs: data_train, ys: y_data})
        lines=ax.plot(np.array(data[['roll_width' ]]),prediction_value,'r-',lw=1)
        plt.pause(0.1)
        # 输出结果可视化-end
        # prediction_value = sess.run(prediction, feed_dict={xs: data_train, ys: y_data})
        # ax.scatter3D(data[['roll_width']], data[['corect_length']], prediction_value)


    if step == (train_times-1) :
         print(step, sess.run(prediction, feed_dict={xs: data_train,ys: y_data}))
         print(step, sess.run((prediction-ys), feed_dict={xs: data_train, ys: y_data}))
         print(step, sess.run(loss, feed_dict={xs: data_train, ys: y_data}))
writer.close()



