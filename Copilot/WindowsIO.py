import cv2
import numpy as np
from mss import mss
import vgamepad as vg
from pygame.locals import *
import pygame

class WindowsIO:
    def __init__(self, processing):
        print("Initialising Windows I/O...")
        self.sct = mss()
        self.gamepad = vg.VX360Gamepad();
        pygame.init()
        self.screen = pygame.display.set_mode((600, 200))
        self.processing = processing

        # Used for determining whether or not the model should have control
        self.model_control = False

    def video_get(self, preprocess=False):
        print(self.sct.monitors)
        img = self.sct.grab(self.sct._monitors[1])
        img = np.array(img)
        img = cv2.resize(img, (960, 540))[:,:,:3]
        if(preprocess):
            img = self.processing.preprocess_image(img)
        return img
    
    def steering_interface_steer(self, angle):
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                cv2.destroyAllWindows()
                pygame.quit()
                return
        
        angle -= 32768
        self.gamepad.left_joystick(x_value = angle, y_value=0)
        self.gamepad.update()
    
    def steering_interface_get(self):
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                cv2.destroyAllWindows()
                pygame.quit()
                return -999
    
        mousepos = pygame.mouse.get_pos()[0]
        steer = int((mousepos / 600) * 65535)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.model_control = True
        if keys[pygame.K_w]:
            self.model_control = False

        mousepos = pygame.mouse.get_pos()[0]
        steer = int((mousepos / 600) * 65535)

        return steer
    
    def display_message(self, message):
        pygame.display.set_caption(message)