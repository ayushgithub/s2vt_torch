import json
import sys
import os
import h5py
import numpy as np
from scipy.misc import imread, imresize
import string
import caffe

json_root_path = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/raw_json'
outpath = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/dataset.h5'
splits = ['train','test','val']
max_caption_length = 0
total_imgs = 0
vgg_model_path = '/home1/ayushmn/thesis/mymodels/neuraltalk2/model/VGG_ILSVRC_16_layers.caffemodel'
vgg_model_proto = '/home1/ayushmn/thesis/mymodels/neuraltalk2/model/VGG_ILSVRC_16_layers_deploy.prototxt'
caffe.set_mode_gpu()
caffe.set_device(0) 
net = caffe.Net(vgg_model_proto,vgg_model_path,caffe.TEST)
image_mean = np.load('/home1/ayushmn/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_mean('data', image_mean)    # subtract the dataset-mean value in each channel
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_raw_scale('data', 255)

def get_image_features(path_list):
	global net, transformer
	net.blobs['data'].reshape(len(path_list),3,224,224)
	images = np.zeros((len(path_list),3,224,224))
	for i,path in enumerate(path_list):
		images[i] = transformer.preprocess('data', caffe.io.load_image(path))
	# images = np.concatenate(images, axis=0)
	# transformed_images = transformer.preprocess('data', images)
	net.blobs['data'].data[...] = images
	output = net.forward()
	return net.blobs['fc7'].data
	pass

def path_sort_key(s):
	global json_root_path, outpath, splits, max_caption_length, total_imgs
	return int(s.split('_')[-1][:-4])

def prepro_caption(input_list, split):
	global json_root_path, outpath, splits, max_caption_length, total_imgs
	print 'processing captions'
	for i,img in enumerate(input_list):
		# print i
		img['split'] = split
		img['caption_tokens'] = str(img['caption'].encode('utf-8')).lower().translate(None, string.punctuation).strip().split()
		if len(img['caption_tokens']) > max_caption_length:
			max_caption_length = len(img['caption_tokens'])
		img['path_list'].sort(key=path_sort_key)
		if len(img['path_list']) > 60:
			img['path_list'] = img['path_list'][:61]
		total_imgs += len(img['path_list'])
	pass

def encode_captions(input_list, wtoi):
	global json_root_path, outpath, splits, max_caption_length, total_imgs
	N = len(input_list)
	M = max_caption_length
	label_arrays = []
	label_length = np.zeros(N, dtype='uint32')

	for i,img in enumerate(input_list):
		assert len(img['caption_tokens']) > 0, 'error: image has no caption'
		Li = np.zeros((1,max_caption_length), dtype='uint32')
		label_length[i] = len(img['caption_tokens'])
		for k,w in enumerate(img['caption_tokens']):
			Li[0,k] = wtoi[w]

		label_arrays.append(Li)

	L = np.concatenate(label_arrays, axis=0)
	assert L.shape[0] == N, 'error'
	assert np.all(label_length) > 0, 'error'
	return L, label_length
	pass


def build_vocab(input_list):
	global json_root_path, outpath, splits, max_caption_length, total_imgs
	counts = {}
	for img in input_list:
		for w in img['caption_tokens']:
			counts[w] = counts.get(w,0) + 1

	cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
	print 'top words and their counts:'
	print '\n'.join(map(str,cw[:20]))
	total_words = sum(counts.itervalues())
	print 'total words:', total_words

	vocab = counts.keys()
	# vocab.append("UNK")
	return vocab
	pass


def process_images(input_list):
	global total_imgs
	image_start_idx = np.zeros((total_imgs,1),dtype='uint32')
	image_end_idx = np.zeros((total_imgs,1),dtype='uint32')
	image_vgg = np.zeros((total_imgs,4096))
	count = 1
	for i, img in enumerate(input_list):
		print img['clip_name'], len(img['path_list'])
		num = len(img['path_list'])
		image_start_idx[i] = count
		image_end_idx[i] = count + num - 1
		image_vgg[count-1 : count + num - 1] = get_image_features(img['path_list'])
		count += num

	return image_vgg, image_start_idx, image_end_idx
	pass

def print_distribution(input_list):
	count = {}
	for i, img in enumerate(input_list):
		count[len(img['path_list'])] = count.get(len(img['path_list']),0) + 1
		if len(img['path_list']) == 355:
			print img['clip_name']
	for key in sorted(count):
	    print "%s: %s" % (key, count[key])
	pass

def main():
	global json_root_path, outpath, splits, max_caption_length, total_imgs
	final_list = []
	for split in splits:
		input_list = json.load(open(os.path.join(json_root_path, split + '.json')))
		prepro_caption(input_list, split)
		final_list += input_list		
	# print total_imgs

	# path = ['/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/images/test/BIG_MOMMAS_LIKE_FATHER_LIKE_SON_DVS32_frame_5.jpg']
	# path = ['/home1/ayushmn/caffe/examples/images/cat.jpg','/home1/ayushmn/caffe/examples/images/cat.jpg']
	# a = get_image_features(path)
	# print np.shape(a)

	# print_distribution(final_list)


	vocab = build_vocab(final_list)
	itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
	wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

	L, label_length = encode_captions(final_list, wtoi)
	f = h5py.File(outpath,'w')
	f.create_dataset("labels", dtype='uint32', data=L)
	f.create_dataset("label_length", dtype='uint32', data=label_length)

	image_vgg, image_start_idx, image_end_idx = process_images(final_list)
	f.create_dataset("image_vgg", data=image_vgg)
	f.create_dataset("image_start_idx", data=image_start_idx)
	f.create_dataset("image_end_idx",data=image_end_idx)

	out = {}
	out['itow'] = itow
	out['final_list'] = final_list
	json.dump(out, open(outpath[:-2]+'json','w'))
	pass

if __name__ == '__main__':
	main()