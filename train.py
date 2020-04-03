import numpy as np
import os
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2 
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from utils.utils import ImageGenerator
from model import ResNet, pyramid_pooling_module, deconvolution_module
from sklearn.model_selection import train_test_split
from utils.utils import get_radiances, get_optical_thickness, crop_images
from utils.losses import binary_focal_loss, focal_loss
from albumentations import Compose, HorizontalFlip, HueSaturationValue, RandomBrightness, RandomContrast, GaussNoise, ShiftScaleRotate

def train_val_generator(args):
  
  h5files = [file for file in os.listdir(args.h5_dir) if file.endswith('.h5')]
  original_X = get_radiances(args.h5_dir, h5files)
  original_y = get_optical_thickness(args.h5_dir, h5files, num_classes=args.num_classes)

  X_dict = crop_images(original_X, args.input_dims, fname_prefix='data')
  y_dict = crop_images(original_y, args.output_dims, fname_prefix='data')

  X_train_list,X_val_list,y_train_list,y_val_list = train_test_split(list(X_dict.keys()),
                                                                     list(y_dict.keys()),
                                                                     shuffle=True,
                                                                     test_size=args.test_size)
  assert X_train_list==y_train_list,'Image names in X and y are different'
  
  AUGMENTATIONS_TRAIN = Compose([HorizontalFlip(p=0.5),
			         RandomContrast(limit=0.2, p=0.75),
			         RandomBrightness(limit=0.2, p=0.75),
				 GaussNoise(p=0.25),
				 ShiftScaleRotate(p=0.5,rotate_limit=20)])

  train_generator = ImageGenerator(image_list=X_train_list,
                                   image_dict=X_dict,
                                   label_dict=y_dict,
                                   input_shape=args.input_dims,
                                   output_shape=args.output_dims,
                                   num_channels=args.input_channels,
                                   num_classes=args.num_classes,
                                   batch_size=args.batch_size,
				   normalize=args.normalize,
				   augmentation=AUGMENTATIONS_TRAIN,
                                   to_fit=True, augment=args.augment, shuffle=True)
  
  val_generator = ImageGenerator(image_list=X_val_list,
                                 image_dict=X_dict,
                                 label_dict=y_dict,
                                 input_shape=args.input_dims,
                                 output_shape=args.output_dims,
                                 num_channels=args.input_channels,
                                 num_classes=args.num_classes,
                                 batch_size=args.batch_size,
				 normalize=args.normalize,
                                 to_fit=True, augment=False, shuffle=True)
  
  return (train_generator,val_generator)


def PSPNet(input_shape, num_channels, out_shape,
           num_classes, learn_rate):
    
  input_layer = Input((input_shape,input_shape,num_channels))
  resnet_block = ResNet(input_layer)
  spp_block = pyramid_pooling_module(resnet_block, out_shape)
  out_layer = deconvolution_module(concat_layer=spp_block,
                                  num_classes=num_classes,
                                  output_shape=(out_shape,out_shape),
				  activation_fn='softmax')
  
  model = Model(inputs=input_layer,outputs=out_layer)
  
  # add regularization to layers
  regularizer = l2(0.01)
  for layer in model.layers:
      for attr in ['kernel_regularizer']:
          if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

  optimizer = Adam(learning_rate=learn_rate, clipnorm=1.0, clipvalue=0.5)
  
  model.compile(optimizer=optimizer,
                loss=focal_loss,
                metrics=['accuracy'])
  
  print('Model has compiled\n')
  # print('The input shape will be {} and the output of'
  #       ' the model will be {}'.format(model.input_shape[1:],model.output_shape[1:]))
  return model

def train_model(model, model_dir, filename, 
                train_generator, val_generator,
                batch_size, epochs):
  
  checkpoint = ModelCheckpoint(os.path.join(model_dir,filename),
                               save_best_only=True, verbose=1)

  lr = ReduceLROnPlateau(monitor='val_loss',factor=0.8, patience=15, verbose=1)

  stop = EarlyStopping(monitor='val_loss', min_delta=0.08, patience=20, verbose=1, mode='min', restore_best_weights=True)
  
  csv = CSVLogger(filename='{}.csv'.format(os.path.splitext(filename)[0]), separator=',', append=True)
  call_list = [checkpoint, lr]
  print('Model will be saved in directory: {} as {}\n'.format(model_dir, filename))
  model.fit_generator(train_generator,
                      validation_data=val_generator,
                      callbacks=call_list,
                      epochs=epochs,verbose=1)
  
  print('Finished training model. Exiting function ...\n')
  return model
