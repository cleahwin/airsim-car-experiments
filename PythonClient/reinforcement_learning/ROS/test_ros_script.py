import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

def publish_dummy_images():
    rospy.init_node('dummy_image_publisher', anonymous=True)
    image_pub = rospy.Publisher('/camera/color/image_rect_raw', Image, queue_size=10)
    bridge = CvBridge()

    # Load your dummy image
    dummy_image = cv2.imread('path_to_dummy_image.jpg')  # Load your dummy image

    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        # Convert OpenCV image to ROS Image message
        ros_image = bridge.cv2_to_imgmsg(dummy_image, encoding="bgr8")
        image_pub.publish(ros_image)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_dummy_images()
    except rospy.ROSInterruptException:
        pass
