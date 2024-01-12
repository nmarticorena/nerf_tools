"""
@file: get_tf.py

Really simple script that get the 4x4 transform matrix from two tfs
on ros

"""
import rospy
import tf
import spatialmath as sm
import spatialmath.base as smb
import numpy as np

rospy.init_node("get_camera_tf")

t = tf.TransformListener()
rospy.sleep(0.1)
while not rospy.is_shutdown():
    try:
        T = t.lookupTransform("/panda_link0", "/base_link", rospy.Time(0))
        print(T)
        T = sm.SE3.Rt(
            R = smb.q2r(T[1], order="xyzs"),
            t = T[0],
            check = False
        )
        T = T.norm()
        np.savetxt("base_link.txt", T.A)

        rospy.signal_shutdown("Finished")
    except Exception  as error:
        print(error)


