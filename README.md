# Copilot
Lane centering using Tensorflow

## About
Copilot is my attempt at building a self-driving system for racing video games. Currently it can handle autosteering on some levels in BeamNG.drive.

## Safety
This code has no concept of what is a "reasonable" steering movement. There is no way to quickly take control back from the model. Please do not try to run this on anything other than a video game or a *small* R/C car.

## Setup
Copilot requires Windows and Python 3.11. Verson 3.12 will not work, as it does not support the needed libraries.
1. Ensure the python binary is in your system PATH
2. Run the `copilot_install_reqs.bat` file.

## Project Structure
`copilot_drive.bat` - Start the autosteering mode  
`copilot_gather.bat` - Gather data to train on  
`copilot_install_reqs.bat` - Install required libraries  
`copilot_test.bat` - Visualize existing drive data  
`copilot_train_custom.bat` - Train on a custom batch size  
`copilot_train.bat` - Train with default settings  
`Copilot.py` - Main file  
`models/lane_centering_model.h5` - Default lane centering model  
`Copilot/Dataset.py` - Code to handle saving/loading the dataset  
`Copilot/ML.py` - Code to handle machine learning  
`Copilot/Processing.py` - Code to handle preprocessing input images, and postprocessing output steering values  
`Copilot/WindowsIO.py` - Code to handle screen capture and XInput controls on Windows.

## Porting to Self-Driving Cars or Other Operating Systems
By default, all I/O for the model is handled in the `Copilot/WindowsIO.py` file. This can easily be replaced with code for another platform, so long as it implements the following components:  
- `model_control` - a `Boolean` to determine whether or not the model should be able to control the wheel. This will be read by some logic in `Copilot.py`, but should only be set by the platform controller.
- `def video_get(preprocess=False)` - A function that returns either a 960x540 scaled-down image (if `preprocess` is `False`), or a preprocessed image (if `preprocess` is `True`).
- `def steering_interface_steer(angle)` - A function that directly pushes out a specific steering angle (where 0 is all the way left and 65535 is all the way right)
- `def steering_interface_get()` - A function that returns the current position of the steering wheel (where 0 is all the way left and 65535 is all the way right)
- `def display_message(message)` - A function that displays a plaintext message to the user 

Once these are implemented, using them is as simple as changing the `from Copilot import ML, Processing, Dataset, WindowsIO` and `io = WindowsIO.WindowsIO(processing)` lines to include your controller. In the future, an argument will be added to allow selecting between different controllers.

**Please note:** Chances are you will need to retrain the model once it has been ported to a different platform. Run `copilot_gather.bat` to gather a few hours of footage, and use `copilot_train.bat` until it is at an acceptable level of performance.

## License
Copilot is released under the MIT License and comes with ABSOLUTELY NO WARRANTY. See `LICENSE` for more information.