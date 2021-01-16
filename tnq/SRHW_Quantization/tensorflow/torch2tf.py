import onnx
from onnx_tf.backend import prepare
import argparse
import os
from models import SRHW
import torch
import numpy as np
from PIL import Image # required if output needs to be displayed
import tensorflow as tf

parser=argparse.ArgumentParser()

parser.add_argument('--model_dir',default='/workspace/SRHW/ckpt/', #check
        help=" Absolute Pytorch Model Directory Path")
parser.add_argument('--ckpt',default='checkpoint1_1',
        help=" Pytorch Model Checkpoint")
parser.add_argument('--model_name',default='SRHW',
        help=" Model Name")
parser.add_argument('--export_dir',default='conversion_output/',
        help=" export directory for onnx and tf models")
parser.add_argument('--img_dir',default='/workspace/SRDataset/train/LR',
        help=" Image directory")
parser.add_argument('--verbose',default=False,type=bool,
        help=" For detailed log")
args=parser.parse_args()
if __name__=='__main__':
    with torch.no_grad():
        pytorch_model=SRHW(quant=True) # As splitting tf model results in undesirable output
        pytorch_model.load_state_dict(torch.load(os.path.join(args.model_dir,args.ckpt+'.pt')))
        dummy_input=torch.randn(1,1,64,64)
        onnx_export=args.export_dir+args.model_name+'.onnx'
        torch.onnx.export(pytorch_model,dummy_input,onnx_export,input_names=["LR"],output_names=["ConvPS"])
        model=onnx.load(onnx_export)
        tf_rep=prepare(model)
        print("-------------------------------------------------------------------------------------")
        print('inputs:',tf_rep.inputs)
        print('outputs:',tf_rep.outputs)
        print("-------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------")
        if args.verbose:
            print('tensor_dict:')
            for element in tf_rep.tensor_dict.items():
                print(element)
            print("-------------------------------------------------------------------------------------")
        #test model
        img= Image.open(os.path.join(args.img_dir,'1.png'))
        img=img.convert('YCbCr')
        img,_,_=img.split()
        img_array=np.asarray(img,dtype=np.float32)
        if img_array.shape[0]>img_array.shape[1]:
            img_array=img_array.transpose(1,0)
        img_array=img_array[np.newaxis,np.newaxis,:,:]
        print("Min:",img_array.min(),"Max:",img_array.max())
        img_array/=255.0
        print("-------------------------------------------------------------------------------------")
        print('input_shape:',img_array.shape)
        output=tf_rep.run(img_array)
        output=np.asarray(output)
        print('output shape:',output.shape)
        print("Min:",output.min(),"Max:",output.max())
        print("-------------------------------------------------------------------------------------")
        
        #freeze the model
        with tf.Session() as sess:
            #graph_def=tf_rep.graph.as_graph_def()
            ######################################################################################3
            saver = tf.compat.v1.train.Saver()
            saver.save(sess,args.export_dir,'saver_'+args.model_name+'.ckpt')
            tf.io.write_graph(graph_def,export_dir,'saver_'+args.model_name+'.pb')
            
            input_ckpt=tf.train.get_checkpoint_state(args.export_dir).model_checkpoint_path
            freeze=tf.train.import_meta_graph(input_ckpt+'.meta',clear_devices=True)
            freeze.restore(sess,input_ckpt)
            ######################################################################################3 
            output_node_names=tf_rep.outputs
            meaningful_names = {}
            
            for output_name in tf_rep.outputs:
                meaningful_names[tf_rep.tensor_dict[output_name].name.replace(':0', '')] = output_name
            for node in graph_def.node:
                if node.name in meaningful_names.keys():
                    node.name = meaningful_names[node.name]
            
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                output_node_names # The output node names are used to select the usefull nodes
            )
            output_graph = os.path.join(args.export_dir,'frozen_'+args.model_name+'.pb') 
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
            if args.verbose:
                for node in graph_def.node:
                    print(node.name)
        
        tf_rep.export_graph(os.path.join(args.export_dir,args.model_name+'.pb'))
        print("-------------------------------------------------------------------------------------")
        print("-->Model Coversion and Freeze Complete<--")

