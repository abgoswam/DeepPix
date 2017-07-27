import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--z_dim', type=int, default=100,
					   help='Noise Dimension')

	parser.add_argument('--t_dim', type=int, default=256,
					   help='Text feature dimension')

	parser.add_argument('--image_size', type=int, default=64,
					   help='Image Size')

	parser.add_argument('--gf_dim', type=int, default=64,
					   help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=64,
					   help='Number of conv in the first layer discr.')

	parser.add_argument('--gfc_dim', type=int, default=1024,
					   help='Dimension of gen untis for for fully connected layer 1024')

	parser.add_argument('--caption_vector_length', type=int, default=2400,
					   help='Caption Vector Length')
	
	parser.add_argument('--data_dir', type=str, default="Data",
					   help='Data Directory')

	parser.add_argument('--model_path', type=str, default='Data/Models/latest_model_flowers_temp.ckpt',
                       help='Trained Model Path')

	parser.add_argument('--n_images', type=int, default=5,
                       help='Number of Images per Caption')

	parser.add_argument('--caption_thought_vectors', type=str, default='Data/sample_caption_vectors.hdf5',
                       help='Caption Thought Vector File')

	
	args = parser.parse_args()
	model_options = {
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.n_images,
		'image_size' : args.image_size,
		'gf_dim' : args.gf_dim,
		'df_dim' : args.df_dim,
		'gfc_dim' : args.gfc_dim,
		'caption_vector_length' : args.caption_vector_length
	}

	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1.")
	gan = model.GAN(model_options)

	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.")
	_, _, _, _, _ = gan.build_model()

	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3.")

	
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4.: {0}".format(args.model_path))
    
#	saver = tf.train.Saver()
#	saver.restore(sess, args.model_path)

	sess = tf.Session()
	saver = tf.train.import_meta_graph(args.model_path)
#	init = tf.local_variables_initializer()
#	sess = tf.Session()
#	sess.run(init) 	
    
	checkpoint_path = os.path.dirname(args.model_path)
	saver.restore(sess,tf.train.latest_checkpoint(checkpoint_path))
	print("TF version : ", tf.__version__)
	for x in tf.global_variables():
		print(x)
		print(sess.run(x))
         
	graph = tf.get_default_graph()
#    self.predict = graph.get_tensor_by_name('predict:0')

#	graph.clear_collection("queue_runners")
#	graph.clear_collection("local_variables")

	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5.")	
	input_tensors, outputs = gan.build_generator()

	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> 6.")
	h = h5py.File( args.caption_thought_vectors )
	caption_vectors = np.array(h['vectors'])
	caption_image_dic = {}
    
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7.")
	for cn, caption_vector in enumerate(caption_vectors):
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7a {0} {1}.".format(cn, caption_vector))
		
		caption_images = []
		z_noise = np.random.uniform(-1, 1, [args.n_images, args.z_dim])
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7b {0}.".format(z_noise.shape))
        
		caption = [ caption_vector[0:args.caption_vector_length] ] * args.n_images
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7c {0}.".format(caption))		
		

            
		[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {
				input_tensors['t_real_caption'] : caption,
				input_tensors['t_z'] : z_noise,
			} )
        
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7d {0} {1}.".format(1,2))		
        
		caption_images = [gen_image[i,:,:,:] for i in range(0, args.n_images)]
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7e {0}.".format(caption_images))
		
		caption_image_dic[ cn ] = caption_images
		print("Generated", cn)

	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 8.")
	for f in os.listdir( join(args.data_dir, 'val_samples')):
		if os.path.isfile(f):
			os.unlink(join(args.data_dir, 'val_samples/' + f))

	for cn in range(0, len(caption_vectors)):
		caption_images = []
		for i, im in enumerate( caption_image_dic[ cn ] ):
			# im_name = "caption_{}_{}.jpg".format(cn, i)
			# scipy.misc.imsave( join(args.data_dir, 'val_samples/{}'.format(im_name)) , im)
			caption_images.append( im )
			caption_images.append( np.zeros((64, 5, 3)) )
		combined_image = np.concatenate( caption_images[0:-1], axis = 1 )
		scipy.misc.imsave( join(args.data_dir, 'val_samples/combined_image_{}.jpg'.format(cn)) , combined_image)


if __name__ == '__main__':
	main()
