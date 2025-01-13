from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

print("required libraries are imported")

print("*******************************************************************")

print("Loading the data")
image_size = (299, 299)
DATA_DIR = './Chessmen_Images_Data'
image_size = image_size
batch_size = 32
print("*******************************************************************")


print("Splitting the data and data augmentation for training data.")

# Data augmentation for training
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,            # Rotate images randomly up to 30 degrees
    width_shift_range=0.2,        # Shift images horizontally by 20% of the width
    height_shift_range=0.2,       # Shift images vertically by 20% of the height
    validation_split=0.2          # Split 20% for validation
)

# No data augmentation for validation

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  # Split 20% for validation
)

# Training data
train_ds = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=image_size,
    batch_size=32,
    subset='training',  # Get the training data
    shuffle=True        # Shuffle training data
)

# Validation data
val_ds = val_gen.flow_from_directory(
    DATA_DIR,
    target_size=image_size,
    batch_size=32,
    subset='validation',  # Get the validation data
    shuffle=False         # Typically, no need to shuffle validation data
)

print("*******************************************************************")

def create_vgg19_model(input_shape=(299, 299, 3), num_classes=6, learning_rate=0.0001, dropout_rate=0.4, freeze_layers = 8):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model
    base_model.trainable = True

    # Freeze all layers except the last few (to fine-tune only the last layers)
    for layer in base_model.layers[:-freeze_layers]:  # Unfreeze the last k layers
        layer.trainable = False

    model = models.Sequential([
        base_model,  # Add VGG19 as base model (without top layer)
        layers.GlobalAveragePooling2D(),  # Pooling layer to reduce spatial dimensions
        layers.Dropout(dropout_rate),  # Add dropout layer after pooling for regularization
        layers.Dense(512, activation='relu'),  # Fully connected layer with ReLU activation
        layers.Dropout(dropout_rate),  # Dropout after the dense layer
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
    ])

    # Compile the model with custom learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),  # Set the learning rate here
        loss='categorical_crossentropy',  # Multi-class classification
        metrics=['accuracy']
    )

    return model

# learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.5,          # Reduce learning rate by a factor of 0.5
    patience=3,          # Wait for 3 epochs before reducing
   
    min_lr=1e-6          # Minimum learning rate
)
print("*******************************************************************")

# model checkpoint callback

checkpoint = ModelCheckpoint(
    'VGG19_v1_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor='val_accuracy',  # Monitoring validation accuracy
    mode='max'            # Saving when val_accuracy improves
)
print("*******************************************************************")

print("Creating the VGG19 model")

model = create_vgg19_model(input_shape=(299, 299, 3), num_classes=6, learning_rate=0.0001, dropout_rate= 0.4, freeze_layers = 8)

print(model.summary())
print("*******************************************************************")

print("Training the VGG19 model")

history = model.fit(
    train_ds,
    epochs=25,
    validation_data=val_ds,
    callbacks=[lr_scheduler, checkpoint]
  )

print("Training of VGG19 model is successfully completed!!!")
print("*******************************************************************")
