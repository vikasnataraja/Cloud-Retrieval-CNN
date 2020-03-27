import numpy as np
import os
import argparse
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
from keras.models import load_model
from train import PSPNet, train_val_generator, train_model


def args_checks_reports(args):

  if not os.path.isdir(args.model_dir):
    print('Model directory {} does not exist,'\
          ' creating it now ...'.format(args.model_dir))
    os.makedirs(args.model_dir)
  
  print('Input dimensions are ({},{},{})\n'.format(args.input_dims, args.input_dims, args.input_channels))
  print('Output dimensions are ({},{},{})\n'.format(args.output_dims, args.output_dims, args.num_classes))
  print('Batch size is {}, learning rate is set '\
        'to {}'.format(args.batch_size,args.lr))
    
if __name__=='__main__':
    
  parser = argparse.ArgumentParser()

  parser.add_argument('--h5_dir', default='./data', type=str, 
                      help="Path to h5 files directory")
  parser.add_argument('--model_dir', default='./model', type=str, 
                      help="Directory where model will be saved.\n" 
                      "If directory does not exist, one will be created")
  parser.add_argument('--model_name', default='pspnet.h5', type=str, 
                      help="File Name of .h5 file which will contain the weights and saved in model_dir")
  parser.add_argument('--input_dims', default=64, type=int, 
                      help="Input dimension")
  parser.add_argument('--input_channels', default=1, type=int, 
                      help="Number of channels in input images")
  parser.add_argument('--output_dims', default=64, type=int, 
                      help="Output dimension")
  parser.add_argument('--num_classes', default=36, type=int, 
                      help="Number of classes")
  parser.add_argument('--batch_size', default=16, type=int, 
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

  args = parser.parse_args()
  
  # check to see if arguments are valid
  args_checks_reports(args)

  # data generators for training and validation data
  train_gen, val_gen = train_val_generator(args)
  
  # build the model
  model = PSPNet(input_shape=args.input_dims, 
                 num_channels=args.input_channels,
                 out_shape=args.output_dims,
                 num_classes=args.num_classes, 
                 learn_rate=args.lr)
  
  trained_model = train_model(model, model_dir=args.model_dir,
                              filename=args.model_name,
                              train_generator=train_gen,
                              val_generator=val_gen,
                              batch_size=args.batch_size,
                              epochs=args.epochs)
  
