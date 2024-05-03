#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
import numpy as np
import torch

class ImageSteeringNode:
    def __init__(self):
        rospy.init_node('image_steering_node', anonymous=True)
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.steering_pub = rospy.Publisher('/steering_angle', Float32, queue_size=10)
        self.model = self.load_model()  # Load your model here

    def load_model(self):
        # Load your pre-trained model here
        # Example:
        # model = NeighborhoodRealCNN()
        # model.load_state_dict(torch.load('path_to_your_model.pth'))
        # model.eval()
        # return model
        return dinov2_vits14  # Use your DINO model

    def preprocess_image(self, image_msg):
        # Convert ROS Image message to OpenCV image
        cv_image = cv2.imdecode(np.frombuffer(image_msg.data, np.uint8), -1)
        # Preprocess the image (resize, normalize, etc.) according to your model requirements
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Assuming your model expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(cv_image).unsqueeze(0)
        return image_tensor

    def infer_steering_angle(self, tensor_image):
        # Use your pre-trained model to infer the steering angle from the image tensor
        with torch.no_grad():
            output = self.model(tensor_image)
            steering_angle = output.item()  # Extract the float value from the tensor
        return steering_angle

    def image_callback(self, image_msg):
        tensor_image = self.preprocess_image(image_msg)
        if tensor_image is not None:
            steering_angle = self.infer_steering_angle(tensor_image)
            self.steering_pub.publish(Float32(steering_angle))

    def run(self):
        rospy.spin()
