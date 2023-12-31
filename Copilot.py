import cv2
import argparse
import os
import random
import string
import time
import numpy as np

global model, processing, dataset, io
from Copilot import ML, Processing, Dataset, WindowsIO

##############
# VISUALIZER #
##############
global steering_wheel
steering_wheel = cv2.imread("steering_wheel.png")

def display_wheel(name, angle):
    global steering_wheel
    image = processing.rotate_image(steering_wheel, ((angle / 65535) - 0.5) * -int(900/2))
    cv2.imshow(name, image)

def visualize(image, ai_image, real_angle, predicted_angle, waitfor=5):
    global preprocessing
    display_image = image.copy()
    display_image = cv2.resize(display_image, (960, 540))
    display_image = cv2.putText(display_image.copy(), f'Steering Angle: {predicted_angle:.2f}, actual {real_angle:.2f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Lane Image', display_image)
    cv2.imshow('AI View', ai_image)
    display_wheel('Actual Wheel', real_angle)
    display_wheel('Predicted Wheel', predicted_angle)
    cv2.waitKey(waitfor)

##############
# MAIN LOGIC #
##############
def main():
    global model, processing, dataset, io
    
    # Get arguments
    print("Parsing argumnents...")
    parser = argparse.ArgumentParser(description='Lane Centering')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the overall accuracy of the model')
    parser.add_argument('--drive', action='store_true', help='Drive using a trained model')
    parser.add_argument('--gather', action='store_true', help='Gather data')

    parser.add_argument('--data-folder', default='data/training_data', help='Custom data folder')
    parser.add_argument('--model-path', default='models/lane_centering_model.h5', help='Custom model path')
    parser.add_argument('--train-count', default=1000, help='Amount of images to train on')
    args = parser.parse_args()

    if(args.train_count == "choose"):
        args.train_count = int(input("Train count? "))

    args.train_count = int(args.train_count)

    # Setup core components
    dataset = Dataset.Dataset(folder=args.data_folder)
    model = ML.ML(model_path=args.model_path, input_shape=(111,480,1))
    processing = Processing.Processing(model)
    io = WindowsIO.WindowsIO(processing)

    ##############
    # TEST MODEL #
    ##############
    if args.test:
        # Get a list of all DriveIDs
        driveIDList = dataset.get_drives()
        driveIDs = list(driveIDList.keys())
        driveID = None
        
        # Let user choose a drive
        while(driveID == None):
            for i in range(0, len(driveIDs)):
                print(str(i) + ": " + driveIDs[i] + " (" + str(len(driveIDList[driveIDs[i]])) + " frames)")
            
            try:
                chosen = int(input("Drive? "))
                driveID = driveIDs[chosen]
            except:
                print("Invalid choice!")
        
        # Get drive data
        driveData = dataset.get_drive_data(driveID)

        # Display all frames
        for frame in driveData:
            angle = frame[0]
            image = cv2.imread(frame[1])
            test_image = processing.preprocess_image(image)
            processing.smooth_steering(model.predict(test_image))
            display_image = processing.preprocess_image_light(image)
            visualize(display_image, test_image, angle, processing.current_steer)

    #################
    # TRAINING CODE #
    #################
    if args.train:
        # List all files in the data_folder
        files = dataset.get_frames(args.train_count)

        # Load all files
        images = []
        steering_angles = []
        for i in range(0, len(files)):
            filename =  files[i]
            if(i % 17 == 0):
                os.system("cls")
                print("Loading files (" + str(int((i/len(files)) * 100)) + "%)")
            steering_angle = float(filename.split('-')[0]) / 65535.0

            # Load and preprocess the image
            image_path = os.path.join(dataset.folder, filename)
            image = processing.preprocess_image(cv2.imread(image_path))

            images.append(image)
            steering_angles.append(steering_angle)
        images = np.array(images)
        steering_angles = np.array(steering_angles)

        # Train
        model.train(images, steering_angles)

    #########
    # DRIVE #
    #########
    if args.drive:
        while True:
            image = io.video_get(preprocess=True)
            model_steering = processing.smooth_steering(model.predict(image))
            human_steering = io.steering_interface_get()

            display_wheel('Wheel', model_steering)

            if(io.model_control):
                io.steering_interface_steer(model_steering)
                io.display_message("Steering: " + str(model_steering))
            else:
                io.steering_interface_steer(human_steering)
                io.display_message("Steering: " + str(model_steering) + " (deviance " + str(abs(model_steering - human_steering)) + ")")

    ########################
    # GATHER TRAINING DATA #
    ########################
    if args.gather:
        driveID = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        frame = 0

        while True:
            # Get image and steering angle, and steer in game
            image = io.video_get()
            steering_value = io.steering_interface_get()
            io.steering_interface_steer(steering_value)
        
            if(io.model_control):
                io.display_message(driveID + ": Currently saving data")
            else:
                io.display_message(driveID + ": Data collection paused")
                continue
            
            image_id = driveID + "-" + str(frame)
            filename = str(steering_value) + "-" + image_id + ".png"
            image_path = os.path.join(dataset.folder, filename)
            cv2.imwrite(image_path, image)
            frame += 1

if __name__ == '__main__':
    print("Copilot LaneKeep running as standalone application")
    main()