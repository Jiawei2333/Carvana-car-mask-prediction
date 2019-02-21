import numpy as np
from PIL import Image
from keras.utils import Sequence
from skimage import transform as sktr
import random


class DataGenerator(Sequence):
    '''
    This is a data generator which output a batch of (X, y) data to feed into keras model.fit_generator method.
    Following inputs are required.
    
        dic_Im_names: defines the unique image names for the training/validation/test sets 
                        with 'train', 'val', and 'test' as the keys, respectively.
        filepath: path of the resized image of size = img_sz. Default value is 'preprocessed/', 
        datatype: indicates the group of the data and takes values of 'train', 'val' or 'test'. Default value is 'train'
        batch_size: batch size. Default set to 16.
        image_sz: size of the input image and mask.Default set to 512
        Im_channels: number of channels of the input image. Default set to 3. 
        msk_channels: number of channels of the mask. Default set to 1. 
        shuffle: shuffle the data for each individual epoch. Default is True.
        aug_on: implement random augmentation (rotation and horizontal flip) for image and mask. default is False.
    '''
    def __init__(self, dic_Im_names,
                 filepath='preprocessed/', 
                 datatype='train', 
                 batch_size=16, 
                 image_sz=512, 
                 Im_channels=3, 
                 msk_channels=1, 
                 shuffle=True, 
                 aug_on=False):
        '''
        init method taking inputs and defines instance varables.
        '''
        self.dic_Im_names = dic_Im_names
        # Select a proper pool (train/val/test) of car names based on the conditions given.
        if datatype == 'train':
            self.Im_names = dic_Im_names['train']
        elif datatype == 'val':
            self.Im_names = dic_Im_names['val']
        elif datatype == 'test':
            self.Im_names = dic_Im_names['test']
        else:
            raise ValueError('Invalid datatype is provided!')    
            
        self.filepath = filepath
        self.batch_size = batch_size
        self.image_sz = image_sz
        self.Im_channels = Im_channels
        self.msk_channels = msk_channels
        self.shuffle = shuffle
        self.aug_on = aug_on
        self.on_epoch_end()        
    
    def __getitem__(self, index):
        '''
        Main function to generate a batch of (X, y) training samples.
        Index defines the current batch index in a epoch.
        '''
        # batch_indexs is a list containing all the indexs of instances in a epoch.
        batch_indexs = self.idxs[index*self.batch_size:(index+1)*self.batch_size]
        # batch_Im_names is a list containing all the image names in the selected batch.
        batch_Im_names = [self.Im_names[i] for i in batch_indexs]
        # X and y are empty array reserved to fed in the instance values (X[i], y[i]).
        X = np.empty((self.batch_size, self.image_sz, self.image_sz, self.Im_channels))
        y = np.empty((self.batch_size, self.image_sz, self.image_sz))
        
        for i, Im_name in enumerate(batch_Im_names):
            X[i] = np.array(Image.open(self.filepath + 'train/' + Im_name))
            y[i] = np.array(Image.open(self.filepath + 'train_masks/' + Im_name.split('.')[0] + '_mask.gif'))
            if self.aug_on:
                X[i], y[i] = self.random_rotation_flip(X[i], y[i])
        X = X/255.0 # Normalization
        # Original y.shape: (self.batch_size, np.shape(y)[1], np.shape(y)[2]). Add one channel dimention to fit in the U-net.
        y = y.reshape(self.batch_size, np.shape(y)[1], np.shape(y)[2], self.msk_channels)
        return X, y
        
    def __len__(self):
        '''
        Return the total number of batches in one epoch.
        '''
        return int(np.floor(len(self.Im_names)/self.batch_size))

    def on_epoch_end(self):
        '''
        Assign indexs to each instance in a epoch.
        Randomly shuffle the index on epoch end, if self.shuffle == True.
        '''
        self.idxs = np.arange(len(self.Im_names))
        if self.shuffle:
            np.random.shuffle(self.idxs)
       
    def random_rotation_flip(self, image, msk):
        '''
        Rotate the image with a random degree smaller than 25.
        '''
        random_degree = random.uniform(-25, 25)
        Image_agmt = sktr.rotate(image, random_degree, mode='symmetric')
        msk_agmt =sktr.rotate(msk, random_degree, mode='symmetric')
        # 50% chance of flipping the image.
        a = random.uniform(0, 1)
        if a > 0.5:
            return Image_agmt[:, ::-1, :], msk_agmt[:, ::-1]
        else:
            return Image_agmt, msk_agmt