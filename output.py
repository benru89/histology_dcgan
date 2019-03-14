import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from dcgan import discriminator, generator, lrelu 
from constants import BASE_PATH, DATA_PATH, CHKPTS_PATH, OUTPUT_PATH, SEED

def save_output(sess,z_batch_tensor,input_z,is_training,steps,dim1,dim2,dim3):
    example_z = sess.run(z_batch_tensor)

    samples, _ = sess.run(generator(input_z, 0.5 , False),feed_dict = {input_z: example_z, is_training:False})
    imgs = [img[:,:,:] for img in samples]
    
    fig, ax = plt.subplots(nrows=4,ncols=4, figsize=(40,40))
    k=0
    for i in range(4):
        for j in range(4):
            ax[i,j].imshow(np.reshape(imgs[k], (dim1,dim2,dim3)), interpolation='nearest')
            ax[i,j].axis('off')
            k = k+1
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    if not os.path.exists(BASE_PATH + OUTPUT_PATH):
        os.mkdir(BASE_PATH + OUTPUT_PATH)    
    fig.savefig(BASE_PATH + OUTPUT_PATH + str(steps) + '_output.png')  
