import os
import numpy as np
import argparse
from keras.optimizers import Adam
from keras.regularizers import l1, l2 
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from albumentations import Compose, HorizontalFlip, HueSaturationValue, RandomBrightness, RandomContrast, GaussNoise, ShiftScaleRotate
from sklearn.model_selection import train_test_split
from architectures.unet import UNet
from architectures.pspnet import PSPNet
from utils.utils import ImageGenerator
from utils.losses import binary_focal_loss, focal_loss, jaccard_distance_loss

def train_val_generator(args):
  X_dict = np.load('{}'.format(args.input_file),allow_pickle=True).item()
  y_dict = np.load('{}'.format(args.ground_truth_file),allow_pickle=True).item()
  assert list(X_dict.keys())==list(y_dict.keys()),'Image names of X and y are different'
  X_train, X_val = train_test_split(list(X_dict.keys()),shuffle=True,random_state=42, test_size=args.test_size)

  AUGMENTATIONS_TRAIN = Compose([HorizontalFlip(p=0.5),
			         RandomContrast(limit=0.2, p=0.5),
			         RandomBrightness(limit=0.2, p=0.5),
				 GaussNoise(p=0.25),
				 ShiftScaleRotate(p=0.5,rotate_limit=20)])

  train_generator = ImageGenerator(image_list=X_train,
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
  
  val_generator = ImageGenerator(image_list=X_val,
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


def build_model(input_shape, num_channels, output_shape, num_classes, learn_rate, loss_fn):
  
  #  model = PSPNet(input_shape, num_channels, output_shape, num_classes, 
  #    		   spatial_pool_sizes=[1,2,3,4], final_activation_fn='softmax',
  #		   transpose=True)   

  model = UNet(input_shape, num_channels, num_classes, final_activation_fn='softmax')
  # add regularization to layers
  regularizer = l2(0.01)
  for layer in model.layers:
    for attr in ['kernel_regularizer']:
      if hasattr(layer, attr):
        setattr(layer, attr, regularizer)

  optimizer = Adam(learning_rate=learn_rate, clipnorm=1.0, clipvalue=0.5)
  print('Loss function being used is: {} loss'.format(loss_fn))
  custom_loss = ''
  if loss_fn == 'focal':
    custom_loss = focal_loss
  elif loss_fn == 'binary_focal':
    custom_loss = binary_focal_loss
  elif loss_fn == 'jaccard':
    custom_loss = jaccard_distance_loss
  elif loss_fn == 'crossentropy':
    custom_loss = 'categorical_crossentropy'

  model.compile(optimizer=optimizer,
                loss=custom_loss,
                metrics=['accuracy'])
  print(model.summary()) 
  print('Model has compiled\n')
  return model

def train_model(model, model_dir, filename, 
                train_generator, val_generator,
                batch_size, epochs):
  
  checkpoint = ModelCheckpoint(os.path.join(model_dir,filename),
                               save_best_only=True, verbose=1)

  lr = ReduceLROnPlateau(monitor='val_loss',factor=0.8, patience=15, verbose=1)
  csv = CSVLogger(filename='{}.csv'.format(os.path.splitext(filename)[0]), separator=',', append=True)
  call_list = [checkpoint, lr]
  print('Model will be saved in directory: {} as {}\n'.format(model_dir, filename))
  model.fit_generator(train_generator,
                      validation_data=val_generator,
                      callbacks=call_list,
                      epochs=epochs,verbose=1)
  
  print('Finished training model. Exiting function ...\n')
  return model

def args_checks_reports(args):
  """ Function to check and print command line arguments """
  if not os.path.isdir(args.model_dir):
    print('Model directory {} does not exist,'\
          ' creating it now ...'.format(args.model_dir))
    os.makedirs(args.model_dir)
  # append .h5 to model_name if it does not have that extension already
  if not os.path.splitext(args.model_name)[1]:
    args.model_name = args.model_name + '.h5'
  print('Input dimensions are ({},{},{})\n'.format(args.input_dims, args.input_dims, args.input_channels))
  print('Output dimensions are ({},{},{})\n'.format(args.output_dims, args.output_dims, args.num_classes))
  print('Batch size is {}, learning rate is set '\
        'to {}'.format(args.batch_size,args.lr))
    
if __name__=='__main__':
    
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', default='data/single_channel/input_radiance.npy', type=str, 
                      help="Path to numpy input images file")
  parser.add_argument('--ground_truth_file', default='data/single_channel/output_cot.npy', type=str,
                      help="Path to numpy ground truth file")
  parser.add_argument('--model_dir', default='weights/', type=str, 
                      help="Directory where model will be saved.\n" 
                      "If directory does not exist, one will be created")
  parser.add_argument('--model_name', default='pspnet.h5', type=str, 
                      help="File Name of .h5 file which will contain the model and saved in model_dir")
  parser.add_argument('--input_dims', default=64, type=int, 
                      help="Input dimension")
  parser.add_argument('--input_channels', default=1, type=int, 
                      help="Number of channels in input images")
  parser.add_argument('--output_dims', default=64, type=int, 
                      help="Output dimension")
  parser.add_argument('--num_classes', default=36, type=int, 
                      help="Number of classes")
  parser.add_argument('--batch_size', default=32, type=int, 
                      help="Batch size for the model")
  parser.add_argument('--lr', default=1e-3, type=float, 
                      help="Learning rate for the model")
  parser.add_argument('--epochs', default=500, type=int, 
                      help="Number of epochs to train the model")
  parser.add_argument('--normalize', default=True, type=bool,
		      help="Flag, set to True if input images need to be normalized")
  parser.add_argument('--augment', default=False, type=bool,
                      help="Flag, set to True if data augmentation needs to be enabled")
  parser.add_argument('--test_size', default=0.20, type=float, 
                      help="Fraction of training image to use for validation during training")
  parser.add_argument('--loss', default='focal', type=str,
		      help="Loss function")
  args = parser.parse_args()
  
  # check to see if arguments are valid
  args_checks_reports(args)

  # data generators for training and validation data
  train_gen, val_gen = train_val_generator(args)
  
  # build the model
  model = build_model(input_shape=args.input_dims, 
                      num_channels=args.input_channels,
                      output_shape=args.output_dims,
                      num_classes=args.num_classes, 
                      learn_rate=args.lr,
		      loss_fn=args.loss)
  
  trained_model = train_model(model, 
			      model_dir=args.model_dir,
                              filename=args.model_name,
                              train_generator=train_gen,
                              val_generator=val_gen,
                              batch_size=args.batch_size,
                              epochs=args.epochs)
