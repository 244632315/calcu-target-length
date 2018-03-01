import tensorflow as tf
import numpy as np
import pandas as pd

print ('hello world!')


data=pd.read_csv("data.csv")


data_train=np.array(data[['roll_length','d_cut_width_i','i_length_slot','hcrop','tcrop']])
y_data=np.array(data[['corect_length']])
data_train=data_train.T
y_data=y_data.T
#print(data_train,data_train.shape)
#print(y_data,y_data.shape)

# print(data_train[0])
# print(data_train[3])
# print(data_train[4])
# print ((data_train[1]*(data_train[2]-1)))
#批量注释ctrl+/



##create TensorFlow structure start
len_crop=tf.Variable(tf.random_uniform([1], -1.0, 1.0),name="len_crop")#大小写注意!

target_length=data_train[0]+data_train[3]+data_train[4]+len_crop+(data_train[1]*(data_train[2]-1))
print(target_length)

loss=tf.reduce_mean(tf.square(target_length-y_data),name="loss")
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss,name="train")

##初始化
init=tf.initialize_all_variables()
##create TensorFlow structure end


#执行初始化
sess=tf.Session()
sess.run(init)##very important

writer=tf.summary.FileWriter('./graphs',sess.graph)
for step in range(800):
    sess.run(train)
    if step %2 ==0:
        print(step,sess.run(len_crop))
writer.close()




#filename_queue=tf.train.string_input_producer(['data.csv'])
#reader=tf.TextLineReader()
#key,value=reader.read(filename_queue)
#print(col=tf.decode_csv(value,record_defaults=[[]*50]))