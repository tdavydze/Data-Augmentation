import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd 

dataset = pd.read_csv("TRAIN_manual.csv",header=None)
dataset1 = dataset.iloc[:,0:784].values
dataset2 = dataset.iloc[:,784:1568].values
dataset3 = dataset.iloc[:,1568:2352].values
dataset4 = dataset.iloc[:,2352:3136].values
dataset1=pd.DataFrame(dataset1)
dataset2=pd.DataFrame(dataset2)
dataset3=pd.DataFrame(dataset3)
dataset4=pd.DataFrame(dataset4)

dataset1 = dataset1.values.reshape((-1,28,28)).clip(0,255).astype(np.uint8)
dataset2 = dataset2.values.reshape((-1,28,28)).clip(0,255).astype(np.uint8)
dataset3 = dataset3.values.reshape((-1,28,28)).clip(0,255).astype(np.uint8)
dataset4 = dataset4.values.reshape((-1,28,28)).clip(0,255).astype(np.uint8)

dataset_final = np.zeros((60,28,28,4))
aug_final= np.zeros((300,28,28,4))

dataset_final[0:60,0:28,0:28,0] = dataset1
dataset_final[0:60,0:28,0:28,1] = dataset2
dataset_final[0:60,0:28,0:28,2] = dataset3
dataset_final[0:60,0:28,0:28,3] = dataset4

    
gen = ImageDataGenerator(rotation_range=180,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.35,
                         zoom_range=0.4, 
                         channel_shift_range=10.,
                         horizontal_flip=True)

for j in range(1,61):
    aug_iter = gen.flow(np.expand_dims(dataset_final[j-1,0:28,0:28,0:4],0))
    aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(5)]
    aug_images1 = np.asarray(aug_images)
    m=j*5-5
    n=m+5
    aug_final[m:n,:,:,:] = aug_images1
    

np.save('keras_augmentation_SAT2.npy',aug_final)
