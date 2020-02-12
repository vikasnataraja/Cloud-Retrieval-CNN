import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
import model as mod
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os,sys
from ImageDataGenerator import ImageGenerator

lr = 1e-4
def build_PSPNet_model(input_shape,num_channels,loss_function='mean_squared_error'):
    
    input_layer = Input((input_shape[0],input_shape[1],num_channels))
    resnet_block = mod.ResNet(input_layer)
    spp_block = mod.build_pyramid_pooling_module(resnet_block)
    deconv_layer = mod.add_deconvolution_layer(concat_layer=spp_block)
    model = Model(inputs=input_layer,outputs=deconv_layer)
    
    adam = Adam(learning_rate=lr)
    
    model.compile(optimizer=adam,
                  loss=loss_function,
                  metrics=['accuracy'])
    return model

def train_val_generator(train_dir, label_dir, batch_size, test_size):
    
    train_generator = ImageGenerator(image_dir=train_dir,
                                     anno_dir=label_dir,
                                     batch_size=batch_size,
                                     n_test=test_size, mode='train')
    
    val_generator = ImageGenerator(image_dir=train_dir,
                                   anno_dir=label_dir,
                                   batch_size=batch_size,
                                   n_test=test_size, mode='valid')
    
    return (train_generator,val_generator)

def train_model(model, filepath, train_generator, val_generator,
                epochs=25, steps_per_epoch=50):
    
    checkpoint = ModelCheckpoint(filepath,save_best_only=False, verbose=1, period=1)
    
    print('Model will be saved in' 
          ' directory: {} as {}\n'.format(os.path.split(filepath)[0],os.path.split(filepath)[1]))
    
    model.fit_generator(train_generator,
                        validation_data=val_generator,
                        callbacks=[checkpoint],
                        epochs=epochs,verbose=1,
                        steps_per_epoch=steps_per_epoch)
    
    print('Finished training model. Exiting function ...\n')
    

