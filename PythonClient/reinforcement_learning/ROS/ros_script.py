import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms_v2
from model import NeighborhoodRealCNN

class SteeringAngleNode:
    def __init__(self):
        rospy.init_node('image_steering_node', anonymous=True)
        # Subscribe to dummy image topic for testing
        self.image_sub = rospy.Subscriber('/camera/color/image_rect_raw', Image, self.image_callback)
        self.steering_pub = rospy.Publisher('/steering_angle', Float32, queue_size=10)
        self.model = self.load_model()  # Load your model here

    def load_model(self):
        model = NeighborhoodRealCNN()
        model.load_state_dict(torch.load('1-2024-05-03.pth'))
        model.eval()
        return model

    def preprocess_image(self, cv_image):
        # Preprocesses image to be compatible with model
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((144, 256)),
            transforms.ToTensor(),
            transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image_tensor = transform(cv_image).unsqueeze(0)
        return image_tensor

    def infer_steering_angle(self, tensor_image):
        # Returns model output given 
        with torch.no_grad():
            output = self.model(tensor_image)
            steering_angle = output.item()  # Extract the float value from the tensor
        return steering_angle

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        tensor_image = self.preprocess_image(cv_image)
        if tensor_image is not None:
            steering_angle = self.infer_steering_angle(tensor_image)
            self.steering_pub.publish(Float32(steering_angle))

    def run(self):
        self.bridge = CvBridge()
        rospy.spin()

def main():
    node = SteeringAngleNode()
    node.run()

if __name__ == '__main__':
    main()