import os
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,DepthwiseConv2D,ReLU,add

tf.keras.backend.set_learning_phase(0)

def get_model(upscale=2,quant=True):
    inputs=tf.keras.Input(shape=(None,None,1),name='LR')
    x=Conv2D(32,3,padding="same",
            use_bias=False,
            #data_format='channels_first',
                          #kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None),
                          #trainable=False
            )(inputs)
    
    x=ReLU(
            #trainable=False
            )(x)
    res=DepthwiseConv2D((1,5),padding='same',
                                    #data_format='channels_first',
                                    use_bias=False,
                                    #depthwise_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                                    #trainable=False
                       )(x)
    res=Conv2D(16,1,use_bias=False,padding='same',
                           #data_format='channels_first',
                           #kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                           #trainable=False
              )(res)
    res=ReLU(
            #trainable=False
            )(res)
    res=DepthwiseConv2D((1,5),padding='same',
                                    #data_format='channels_first',
                                    use_bias=False,
                                    #depthwise_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                                    #trainable=False
                                    )(res)
    res=Conv2D(32,1,use_bias=False,padding='same',
                           #data_format='channels_first',
                           #kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                           #trainable=False
              )(res)
    res=ReLU()(res)
    
    x=add([x,res],
            #trainable=False
            )
    x=DepthwiseConv2D(3,padding='same',
                                    #data_format='channels_first',
                                    use_bias=False,
                                    #depthwise_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                                    #trainable=False
                     )(x)
    x=Conv2D(16,1,use_bias=False,padding='same',
                           #data_format='channels_first',
                           #kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                           #trainable=False
            )(x)
    x=ReLU(
            #trainable=False
           )(x)
    x=DepthwiseConv2D(3,padding='same',
                                    #data_format='channels_first',
                                    use_bias=False,
                                    #depthwise_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                                    #trainable=False
                     )(x)
    x=Conv2D(upscale**2,1,use_bias=False,padding='same',
                           #data_format='channels_first',
                           #kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)
                           #trainable=False
            )(x)
    ps=tf.nn.depth_to_space(x,upscale,data_format='NHWC',name='HR')
    model=None
    if quant:
        model=tf.keras.Model(inputs=inputs,outputs=x)
    else:
        model=tf.keras.Model(inputs=inputs,outputs=ps)
    model.summary()
    return model

def keras_convert(keras_json,keras_hdf5,tf_ckpt):

    ##############################################
    # load the saved Keras model
    ##############################################

    # set learning phase for no training
    tf.keras.backend.set_learning_phase(0)
    
    # if name of JSON file provided as command line argument, load from 
    # arg.keras_json and args.keras_hdf5.
    # if JSON not provided, assume complete model is in HDF5 format
    
    model=get_model(quant=True)
    model.load_weights(keras_hdf5)
    for layer in model.layers:
        print(layer.trainable)
    ##############################################
    # Create TensorFlow checkpoint & inference graph
    ##############################################

    print ('Keras model information:')
    print (' Input names :',model.input.op.name)
    print (' Output names:',model.output.op.name)
    print('-------------------------------------')


    # fetch the tensorflow session using the Keras backend
    tf_session = keras.backend.get_session()


    # write out tensorflow checkpoint & meta graph
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(tf_session,tf_ckpt)
    print (' Checkpoint created :',tf_ckpt)

    return



def run_main():

    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-kj', '--keras_json',
                    type=str,
                    default='',
    	            help='path of Keras JSON. Default is empty string to indicate no JSON file')
    ap.add_argument('-kh', '--keras_hdf5',
                    type=str,
                    default='./model.hdf5',
    	            help='path of Keras HDF5. Default is ./model.hdf5')
    ap.add_argument('-tf', '--tf_ckpt',
                    type=str,
                    default='./tf_float.ckpt',
    	            help='path of TensorFlow checkpoint. Default is ./tf_float.ckpt')           
    args = ap.parse_args()

    print('-------------------------------------')
    print('keras_2_tf command line arguments:')
    print(' --keras_json:', args.keras_json)
    print(' --keras_hdf5:', args.keras_hdf5)
    print(' --tf_ckpt   :', args.tf_ckpt)
    print('-------------------------------------')

    keras_convert(args.keras_json,args.keras_hdf5,args.tf_ckpt)



if __name__ == '__main__':
    run_main()

