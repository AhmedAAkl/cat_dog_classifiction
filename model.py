# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('th')
import time

from numpy.random import seed

# source of trian and validation dataset
train_data_dir = ""
valid_data_dir = ""

# output directory to save model ouput weights
output_dir = ""        
    



def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(3, 72, 72),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Conv2D(128, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3),padding='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1,activation='sigmoid'))
    
    opt = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)    
    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    model.summary()
    return model
    

if __name__ == '__main__':
        
    batch_size = 32
    img_rows,img_cols = 72,72
    num_channels = 3
    nb_epochs = 300
    
    seed(22)
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            train_data_dir,  # this is the target directory
            target_size=(img_rows, img_cols),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
    
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            valid_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='binary')
    
    train_images_num = len(train_generator.filenames)
    valid_images_num = len(validation_generator.filenames)
    
    # loading the model
    
    model = create_model()
 
    file_path = output_dir + 'model.h5'
    
    early_stopping = EarlyStopping(monitor='val_acc',patience=30,verbose=0,mode='auto')
    checkpoint = ModelCheckpoint(file_path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list = [checkpoint,early_stopping]
    
    start = time.time()
    
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=train_images_num//batch_size,
            epochs=nb_epochs,
            verbose=2,
            callbacks=callbacks_list,
            validation_data=validation_generator,
            validation_steps=valid_images_num//batch_size)
    end = time.time()
    
    traing_time = end - start
    
    print(str(traing_time))
    scores = model.evaluate_generator(validation_generator,
                             steps=None)
    
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Loss: %.2f%%" % (scores[0]))
