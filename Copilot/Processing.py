import cv2
import numpy as np

class Processing:
    def __init__(self, model, steer_max=65535, steer_alpha = 0.2):
        # For image preprocessing
        self.image_shape = (model.input_shape[1], model.input_shape[0])
        
        # For steering scaling
        self.steer_max = 65535

        # For smooth steering
        self.current_steer = 32768
        self.steer_alpha = steer_alpha

    def blur(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred

    def grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        return gray

    def edge_detection(self, image):
        edges = cv2.Canny(image, 100, 200)
        return edges

    def preprocess_image(self, image):
        image = image[146:368, 0:960]
        image = self.grayscale(image)
        image = self.edge_detection(image)
        image = self.blur(image)
        image = cv2.resize(image, self.image_shape)  # Resize the image to match the model input shape
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image
    
    def preprocess_future(self, image, frame):
        image = image[146 + (frame * 64):368 + (frame * 64), 0:960]
        image = self.grayscale(image)
        image = self.edge_detection(image)
        image = self.blur(image)
        image = cv2.resize(image, self.image_shape)  # Resize the image to match the model input shape
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image

    def preprocess_image_light(self, image):
        image = cv2.resize(image, self.image_shape)  # Resize the image to match the model input shape
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image
    
    def postprocess_steering(self, output):
        steering = output * self.steer_max
        
        if(steering < 0):
            steering = 0

        if(steering > self.steer_max):
            steering = self.steer_max
        
        return int(steering)
    
    def smooth_steering(self, output):
        angle = self.postprocess_steering(output)
        self.current_steer = int(self.steer_alpha * angle + (1 - self.steer_alpha) * self.current_steer)
        return self.current_steer
        
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result