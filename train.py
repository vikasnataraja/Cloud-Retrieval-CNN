import os
import numpy as np
import argparse
from keras.optimizers import Adam
from keras.regularizers import l1, l2 
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from albumentations import Compose, HorizontalFlip, HueSaturationValue, RandomBrightness, RandomContrast, GaussNoise, ShiftScaleRotate
from sklearn.model_selection import train_test_split
from models.unet import UNet, Experimental_UNet
from models.pspnet import PSPNet
from utils.utils import ImageGenerator
from utils.losses import focal_loss, combined_loss

def train_val_generator(args):
  X_dict = np.load('{}'.format(args.input_file),allow_pickle=True).item()
  y_dict = np.load('{}'.format(args.output_file),allow_pickle=True).item()
  assert list(X_dict.keys())==list(y_dict.keys()),'Image names of X and y are different'
  print('Total number of data files available for training = {}\n\n'.format(int((1-args.test_size)*len(X_dict))))
  X_train, X_val = train_test_split(list(X_dict.keys()), shuffle=True, random_state=42, test_size=args.test_size)

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
                                   num_channels=args.num_channels,
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
                                 num_channels=args.num_channels,
                                 num_classes=args.num_classes,
                                 batch_size=args.batch_size,
				 normalize=args.normalize,
                                 augmentation=None,
                                 to_fit=True, augment=False, shuffle=True)
  
  return (train_generator,val_generator)


def build_model(input_shape, num_channels, output_shape, num_classes, learn_rate, fine_tune, path_to_weights):
  
  model = UNet(input_shape, num_channels, num_classes, final_activation_fn='softmax')
  
  # add regularization to layers
  regularizer = l2(0.01)
  for layer in model.layers:
    for attr in ['kernel_regularizer']:
      if hasattr(layer, attr):
        setattr(layer, attr, regularizer)

  optimizer = Adam(learning_rate=learn_rate, clipnorm=1.0, clipvalue=0.5)
  
  if fine_tune:
    model.load_weights(path_to_weights)
    print('Pre-trained weights loaded to model\n')
    for layer in model.layers[:-2]:
      layer.trainable = False
  
  model.compile(optimizer=optimizer,
                loss=focal_loss,
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
                      epochs=epochs,
                      verbose=1,
                      max_queue_size=10,
                      workers=1)
  
  print('Finished training model. Exiting function ...\n')
  return model

def args_checks_reports(args):
  """ Function to check and print command line arguments """
  print('-----------------------------------------------------')
  if not os.path.isdir(args.model_dir):
    print('\n Model directory {} does not exist,'\
          ' creating it now ...'.format(args.model_dir))
    os.makedirs(args.model_dir)

  # append .h5 to model_name if it does not have that extension already
  if not os.path.splitext(args.model_name)[1]:
    args.model_name = args.model_name + '.h5'

  if args.normalize:
    print('\nImages will be normalized\n')
  else:
    print('\nImages will not be normalized\n')

  if args.augment:
    print('Data augmentation will be used\n')
  else:
    print('Data augmentation will not be used\n')

  print('Input dimensions are ({},{},{})\n'.format(args.input_dims, args.input_dims, args.num_channels))
  print('Output dimensions are ({},{},{})\n'.format(args.output_dims, args.output_dims, args.num_classes))
  print('Batch size is {}, learning rate is set '\
        'to {}\n'.format(args.batch_size,args.lr))
  print('-----------------------------------------------------')
    
if __name__=='__main__':
    
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', default='data/single_channel/input_radiance.npy', type=str, 
                      help="Path to numpy input images file")
  parser.add_argument('--output_file', default='data/single_channel/output_cot.npy', type=str,
                      help="Path to numpy ground truth file")
  parser.add_argument('--model_dir', default='weights/', type=str, 
                      help="Directory where model will be saved."
                      "If directory does not exist, one will be created")
  parser.add_argument('--model_name', default='unet.h5', type=str, 
                      help="Model name that will be saved in model_dir")
  parser.add_argument('--input_dims', default=64, type=int, 
                      help="Input dimension")
  parser.add_argument('--num_channels', default=1, type=int, 
                      help="Number of channels in input images, set to 1 by default")
  parser.add_argument('--output_dims', default=64, type=int, 
                      help="Output dimension, set to 64 by default to get 64x64 images")
  parser.add_argument('--num_classes', default=36, type=int, 
                      help="Number of classes")
  parser.add_argument('--batch_size', default=32, type=int, 
                      help="Batch size for the model")
  parser.add_argument('--lr', default=1e-3, type=float, 
                      help="Learning rate for the model")
  parser.add_argument('--epochs', default=500, type=int, 
                      help="Number of epochs to train the model")
  parser.add_argument('--normalize', dest='normalize', action='store_true',
		      help="Pass --normalize to normalize the images. By default, images will not be normalized")
  parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                      help="Pass --fine_tune to load a previous model and fine tune. By default, this is set to False")
  parser.add_argument('--weights_path', default='~/workspace/weights/unet.h5', type=str,
                      help="If fine tuning, pass a path to weights that will be loaded and fine-tuned")
  parser.add_argument('--augment', dest='augment', action='store_true',
                      help="Pass --augment to use data augmentation. By default, no augmentation is used")
  parser.add_argument('--test_size', default=0.20, type=float, 
                      help="Fraction of training image to use for validation during training")
  args = parser.parse_args()
  
  # check to see if arguments are valid
  args_checks_reports(args)

  # data generators for training and validation data
  train_gen, val_gen = train_val_generator(args)
  
  # build the model
  model = build_model(input_shape=args.input_dims,
		      num_channels=args.num_channels,
		      output_shape=args.output_dims,
		      num_classes=args.num_classes,
		      learn_rate=args.lr,
                      fine_tune=args.fine_tune,
                      path_to_weights=args.weights_path)
  
  trained_model = train_model(model, 
			      model_dir=args.model_dir,
                              filename=args.model_name,
                              train_generator=train_gen,
                              val_generator=val_gen,
                              batch_size=args.batch_size,
                              epochs=args.epochs)
