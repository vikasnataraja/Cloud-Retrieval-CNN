import numpy as np
import os
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ImageDataGenerator import ImageGenerator
from model import ResNet, pyramid_pooling_module, deconvolution_module
from sklearn.model_selection import train_test_split
from h5_to_img import get_optical_thickness, get_radiances
from slice_images import crop_images

def train_val_generator(args):
  
  files = [file for file in os.listdir(args.h5_dir) if file.endswith('.h5')]
  original_X = get_radiances(args.h5_dir, files)
  original_y = get_optical_thickness(args.h5_dir, files)

  X_dict = crop_images(original_X, args.input_dims)
  y_dict = crop_images(original_y, args.output_dims)

  X_train_list,X_val_list,y_train_list,y_val_list = train_test_split(list(X_dict.keys()),
                                                                     list(y_dict.keys()),
                                                                     shuffle=False,
                                                                     test_size=args.test_size)
  
  assert len(X_train_list)==len(y_train_list),'Number of images is not equal to number of labels'
  
  train_generator = ImageGenerator(image_list=X_train_list,
                                   label_list=y_train_list,
                                   image_dict=X_dict,
                                   label_dict=y_dict,
                                   input_shape=args.input_dims,
                                   output_shape=args.output_dims,
                                   num_channels=args.input_channels,
                                   num_classes=args.num_classes,
                                   batch_size=args.batch_size,
                                   to_fit=True)
  
  val_generator = ImageGenerator(image_list=X_val_list,
                                 label_list=y_val_list,
                                 image_dict=X_dict,
                                 label_dict=y_dict,
                                 input_shape=args.input_dims,
                                 output_shape=args.output_dims,
                                 num_channels=args.input_channels,
                                 num_classes=args.num_classes,
                                 batch_size=args.batch_size,
                                 to_fit=True)
  
  return (train_generator,val_generator)


def PSPNet(input_shape, num_channels, out_shape,
           num_classes, learn_rate):
    
  print('Started building PSPNet\n')
  input_layer = Input((input_shape,input_shape,num_channels))
  resnet_block = ResNet(input_layer)
  spp_block = pyramid_pooling_module(resnet_block)
  out_layer = deconvolution_module(concat_layer=spp_block,
                                  num_classes=num_classes,
                                  output_shape=(out_shape,out_shape))
  
  model = Model(inputs=input_layer,outputs=out_layer)
  
  optimizer = Adam(learning_rate=learn_rate)
  
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  print('Model has compiled\n')
  print('The input shape will be {} and the output of'
        ' the model will be {}'.format(model.input_shape[1:],model.output_shape[1:]))
  return model


def train_model(model, model_dir, filename, train_generator, val_generator,
                batch_size, epochs):
  
  checkpoint = ModelCheckpoint(os.path.join(model_dir,filename),
                               save_best_only=False, verbose=1)

  print('Model will be saved in' 
        ' directory: {} as {}\n'.format(model_dir, filename))
  
  model.fit_generator(train_generator,
                      validation_data=val_generator,
                      callbacks=[checkpoint],
                      epochs=epochs,verbose=1,
                      steps_per_epoch=np.ceil(len(train_generator)/batch_size))
  
  print('Finished training model. Exiting function ...\n')
  
  return model