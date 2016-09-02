import sys
import os
import json

images_root = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/images'
annotations_root = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/M_VAD/srt_files/'
outpath = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/raw_json'
images_split = ['test','val','train']
annotations_split = ['test_srt','valid_srt','train_srt']

def process_srt(annotations_path):
	out_dict = {}
	file_list = [os.path.join(annotations_path,file) for file in os.listdir(annotations_path)]
	for file_name in file_list:
		with open(file_name) as f:
			count = 1
			for line in f.readlines():
				if count % 4 == 1:
					clip_name = line.strip()
				elif count % 4 == 3:
					clip_caption = line.strip()
					out_dict[clip_name] = clip_caption
				else:
					pass

				count += 1

	return out_dict
	pass


for index, split in enumerate(images_split):
	final_list = []
	images_path = os.path.join(images_root,split)
	annotations_path = os.path.join(annotations_root, annotations_split[index])

	images_list = os.listdir(images_path)
	clip_dict = {}
	clip_list = []
	for image_name_frame in images_list:
		image_name = '_'.join(image_name_frame.strip().split('_')[:-2])
		if image_name in clip_dict:
			clip_dict[image_name].append(os.path.join(images_path,image_name_frame))
		else:
			clip_list.append(image_name)
			clip_dict[image_name]=[]
			clip_dict[image_name].append(os.path.join(images_path,image_name_frame))


	# clip_list = ['_'.join(image_name.strip().split('_')[:-2]) for image_name in images_list]
	# clip_list = list(set(clip_list))
	clip_captions = process_srt(annotations_path)

	for clip in clip_list:
		if clip in clip_captions:
			temp_dict = {'clip_name': clip, 'caption': clip_captions[clip], 'path_list':clip_dict[clip]}
		final_list.append(temp_dict)

	json.dump(final_list,open(os.path.join(outpath,split+'.json',),'w'))
