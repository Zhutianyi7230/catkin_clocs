#!/data/yum_install/usr/bin/python 
# -*- coding: utf-8 -*-
import numpy as np
import std_msgs
import rospy
from std_msgs.msg import Float32MultiArray,Int32
import threading

class PublisherNode():
    def __init__(self):
        rospy.init_node('publisher', anonymous=True)
        self.pub1 = rospy.Publisher('det2d_array', Float32MultiArray, queue_size=10)
        self.pub2 = rospy.Publisher('det3d_array1', Float32MultiArray, queue_size=10)
        self.pub3 = rospy.Publisher('det3d_array2', Float32MultiArray, queue_size=10)
        self.pub4 = rospy.Publisher('rect_array', Float32MultiArray, queue_size=10)
        self.pub5 = rospy.Publisher('Trv2c_array', Float32MultiArray, queue_size=10)
        self.pub6 = rospy.Publisher('P2_array', Float32MultiArray, queue_size=10)
        
        rospy.Subscriber('predictions_result', Float32MultiArray, self.subscriber_callback)
        
        self.condition = threading.Condition()
        self.is_subscriber_ready = False

    def subscriber_callback(self,msg):
        rospy.loginfo("Received message from subscriber.")

        # Notify the waiting publisher
        with self.condition:
            self.is_subscriber_ready = True
            self.condition.notify()

        
    def random_array_publisher(self):

        # 使用NumPy生成随机的Float32MultiArray数据
        data = Float32MultiArray()
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[0].size = 30
        data.layout.dim[0].stride = 6
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[1].size = 6
        data.layout.dim[1].stride = 1
        # 生成数据矩阵
        sample_data = np.load('/media/zhutianyi/KESU/project/catkin_clocs/src/clocs/top_predictions.npy')
        sample_data = sample_data.astype(np.float32)
        data.data = sample_data.flatten().tolist()
        self.pub1.publish(data)
        rospy.loginfo("Publishing random Float32MultiArray data for det2d.")
        
        data = Float32MultiArray()
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[0].size = 20000
        data.layout.dim[0].stride = 7
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[1].size = 7
        data.layout.dim[1].stride = 1
        # 生成数据矩阵
        sample_data = np.load('/media/zhutianyi/KESU/project/catkin_clocs/src/clocs/box_preds_3d.npy')
        sample_data = sample_data.astype(np.float32)
        data.data = sample_data.flatten().tolist()
        self.pub2.publish(data)
        rospy.loginfo("Publishing random Float32MultiArray data for det3d_1.")
        
        data = Float32MultiArray()
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[0].size = 20000
        data.layout.dim[0].stride = 3
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[1].size = 3
        data.layout.dim[1].stride = 1
        # 生成数据矩阵
        sample_data = np.load('/media/zhutianyi/KESU/project/catkin_clocs/src/clocs/cls_preds_3d.npy')
        sample_data = sample_data.astype(np.float32)
        data.data = sample_data.flatten().tolist()
        self.pub3.publish(data)
        rospy.loginfo("Publishing random Float32MultiArray data for det3d_2.")
        
        data = Float32MultiArray()
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[0].size = 4
        data.layout.dim[0].stride = 4
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[1].size = 4
        data.layout.dim[1].stride = 1
        # 生成数据矩阵
        random_data = np.array([[ 9.999128000000e-01, 1.009263000000e-02, -8.511932000000e-03,  -1.012729000000e-02],
                                [ -1.012729000000e-02,  9.999406000000e-01, -4.037671000000e-03,  0.0000],
                                [ 8.470675000000e-03,  4.123522000000e-03,  9.999556000000e-01,  0.0000],
                                [ 0.0000,  0.0000,  0.0000,  1.0000]]).astype(np.float32)
        data.data = random_data.flatten().tolist()
        self.pub4.publish(data)
        rospy.loginfo("Publishing random Float32MultiArray data for rect.")
        
        data = Float32MultiArray()
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[0].size = 4
        data.layout.dim[0].stride = 4
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[1].size = 4
        data.layout.dim[1].stride = 1
        # 生成随机数据矩阵
        random_data = np.array([[ 6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02],
                                [ -1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01, -6.127237000000e-02],
                                [ 9.999753000000e-01,  6.931141000000e-03,  -1.143899000000e-03, -3.321029000000e-01],
                                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).astype(np.float32)
        data.data = random_data.flatten().tolist()
        self.pub5.publish(data)
        rospy.loginfo("Publishing random Float32MultiArray data for Trv2c.")
        data = Float32MultiArray()
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[0].size = 4
        data.layout.dim[0].stride = 4
        data.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        data.layout.dim[1].size = 4
        data.layout.dim[1].stride = 1
        # 生成随机数据矩阵
        random_data = np.array([[7.070493000000e+02, 0.0000e+00, 6.040814000000e+02, 4.575831000000e+01],
                                [0.0000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
                                [0.0000e+00, 0.0000e+00, 1.0000e+00, 4.981016000000e-03],
                                [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]).astype(np.float32)
        data.data = random_data.flatten().tolist()
        self.pub6.publish(data)
        rospy.loginfo("Publishing random Float32MultiArray data for P2.")
    
    def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        self.random_array_publisher()
        
        while not rospy.is_shutdown():
            with self.condition:
                while not self.is_subscriber_ready:
                    # Wait for the subscriber to signal
                    self.condition.wait()

                # Publish a message after receiving the subscriber's signal
                self.random_array_publisher()
                self.is_subscriber_ready = False

            rate.sleep()    
        
        
if __name__=="__main__":
    try:
        custom_node = PublisherNode()
        rospy.sleep(10)#wait for clocs_fusion.py
        custom_node.run()
    except rospy.ROSInterruptException:
        pass

    
    