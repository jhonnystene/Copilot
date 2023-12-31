import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import datetime

class ML:
    def __init__(self, model_path="models/lane_centering_model.h5", input_shape=(144, 480, 3)):
        print("Starting CopilotML with model " + model_path)
        self.model_path = model_path
        self.input_shape = input_shape

        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model(model_path)
            print("Done.")
        except:
            print("Couldn't load model - must not exist. Creating model...")
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=self.input_shape),
                tf.keras.layers.Conv2D(16, (8, 8), strides=(4, 4), padding="same", activation='relu'),
                tf.keras.layers.ELU(),
                tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu'),
                tf.keras.layers.ELU(),
                tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.ELU(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.ELU(),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mae')
            print("Done.")
    
    def train(self, training_images, training_angles, split_ratio=0.8, training_epochs=50):
        num_samples = len(training_images)
        split_index = int(num_samples * split_ratio)
        train_images = training_images[:split_index]
        test_images = training_images[split_index:]
        train_angles = training_angles[:split_index]
        test_angles = training_angles[split_index:]

        checkpoint_callback = ModelCheckpoint(
            filepath="best_model.h5",
            monitor="val_loss",  # Monitor validation loss
            save_best_only=True,  # Save only the best model
            verbose=1
        )

        self.model.fit(train_images, train_angles, epochs=training_epochs, batch_size=64, 
            validation_data=(test_images, test_angles), 
            callbacks=[checkpoint_callback]
        )
        
        self.model = tf.keras.models.load_model("best_model.h5")
        self.save_model()
    
    def save_model(self):
        print("Saving model...")
        self.model.save(self.model_path)
        print("Done.")
    
    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.model.save(datetime.now().strftime("CHECKPOINT-%Y-%m-%d-%H-%M.h5"))
        print("Done.")

    def predict(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))[0][0]
        return prediction
    
    def list_gpus(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            for gpu in gpus:
                print("Name:", gpu.name)
                print("Device:", gpu.device_type)
        else:
            print("No GPU detected.")