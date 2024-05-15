import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms
# from torchvision.transforms import v2 as transforms_v2
from model import NeighborhoodRealCNN
from ackermann_msgs.msg import AckermannDriveStamped
from mavros_msgs.msg import RCIn
import time

class SteeringAngleNode:
    def __init__(self):
        rospy.init_node('image_steering_node', anonymous=True)
        # Subscribe to dummy image topic for testing
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.sub_channel = rospy.Subscriber("/mavros/rc/in", RCIn, self.channel_cb, queue_size=2)

        self.control_pub = rospy.Publisher( "low_level_controller/hound/control", AckermannDriveStamped, queue_size=1)
        self.model = self.load_model()  # Load your model here
        self.throttle_to_wheelspeed = 5.0
        self.steering_max = 0.488
        self.speed = 0
        print("model ready")

    def load_model(self):
        model = NeighborhoodRealCNN()
        model.load_state_dict(torch.load('1-2024-05-03.pth'))
        model.to("cuda")
        model.eval()
        return model

    def preprocess_image(self, cv_image):
        cv_image = (np.float32(cv2.resize(cv_image,(140,252))) - 128.0)/255.0 # normalize and reshape
        input_tensor = torch.tensor(cv_image.transpose((2,1,0))).to("cuda").unsqueeze(0)
        return input_tensor

    def infer_steering_angle(self, tensor_image):
        # Returns model output given 
        with torch.no_grad():
            output = self.model(tensor_image)
            steering_angle = output.cpu().item()  # Extract the float value from the tensor
        return steering_angle

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        tensor_image = self.preprocess_image(cv_image)
        ctrl = np.zeros(2)
        ctrl[0] = self.infer_steering_angle(tensor_image)
        ctrl[1] = self.speed
        self.send_ctrl(ctrl)

    def channel_cb(self, rc):
        try:
            if(len(rc.channels) == 0 ):
                return
            self.speed = (rc.channels[2] - 1000)/1000.0
        except:
            pass

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = ctrl[0] * self.steering_max
        control_msg.drive.speed = ctrl[1] * self.throttle_to_wheelspeed
        self.control_pub.publish(control_msg)

    def run(self):
        rospy.spin()

def main():
    node = SteeringAngleNode()
    node.run()

if __name__ == '__main__':
    main()
