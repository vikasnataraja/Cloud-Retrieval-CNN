import numpy as np
import os
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ImageDataGenerator import ImageGenerator
from model import ResNet, pyramid_pooling_module, deconvolution_module
from sklearn.model_selection import train_test_split


def train_val_generator(args, val_size=0.25):
    
    
    X_train,X_val,y_train,y_val = train_test_split(os.listdir(args.train_dir),
                                                   os.listdir(args.label_dir),
                                                   shuffle=False,
                                                   test_size=val_size)
    assert len(X_train)==len(y_train),'Number of images is not equal to number of labels'
    
    train_generator = ImageGenerator(image_list=X_train,
                                     label_list=y_train,
                                     image_dir=args.train_dir,
                                     anno_dir=args.label_dir,
                                     resize_shape_tuple=args.input_dims,
                                     num_channels=args.num_channels,
                                     num_classes=args.num_classes,
                                     batch_size=args.batch_size)
    
    val_generator = ImageGenerator(image_list=X_val,
                                   label_list=y_val,
                                   image_dir=args.train_dir,
                                   anno_dir=args.label_dir,
                                   resize_shape_tuple=args.input_dims,
                                   num_channels=args.num_channels,
                                   num_classes=args.num_classes,
                                   batch_size=args.batch_size)
    
    return (train_generator,val_generator)


def PSPNet(input_shape, num_channels, out_shape,
           num_classes, learn_rate, loss_function):
    
    print('Started building PSPNet\n')
    input_layer = Input((input_shape[0],input_shape[1],num_channels))
    resnet_block = ResNet(input_layer)
    spp_block = pyramid_pooling_module(resnet_block)
    out_layer = deconvolution_module(concat_layer=spp_block,
                                    num_classes=num_classes,
                                    output_shape=out_shape)
    
    model = Model(inputs=input_layer,outputs=out_layer)
    
    adam = Adam(learning_rate=learn_rate)
    
    model.compile(optimizer=adam,
                  loss=loss_function,
                  metrics=['accuracy'])
    
    print('Model has compiled\n')
    print('The input shape will be {} and the output of'
          ' the model will be {}'.format(model.input_shape[1:],model.output_shape[1:]))
    return model


def train_model(model, model_dir, filename, train_generator, val_generator,
                epochs, steps_per_epoch):
    
    checkpoint = ModelCheckpoint(os.path.join(model_dir,filename),
                                 save_best_only=False, verbose=1)
    
    print('Model will be saved in' 
          ' directory: {} as {}\n'.format(model_dir, filename))
    
    model.fit_generator(train_generator,
                        validation_data=val_generator,
                        callbacks=[checkpoint],
                        epochs=epochs,verbose=1,
                        steps_per_epoch=steps_per_epoch)
    
    print('Finished training model. Exiting function ...\n')
    
    return model