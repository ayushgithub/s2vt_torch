import sys
import os
import cv2
import random

file_list = ['test_split/TestList.txt','train_split/TrainList.txt','valid_split/ValidList.txt']
root_path = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/M_VAD/split/'

video_folder = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/video'

video_list = os.listdir(video_folder)

output_folder = '/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/images'
output_list = ['test','train','val']


def saveframes(vidpath, outpath):
	vidname = vidpath.strip().split("/")[-1][:-4]
	outpath = os.path.join(outpath,vidname)
	vidcap = cv2.VideoCapture(vidpath)
	success,image = vidcap.read()
	count = 1
	while success:
		if count%5 == 0:
			num1 = random.randint(0,32)
			num2 = random.randint(0,32)
			image = cv2.resize(image,(256,256))
			image = image[num1:num1+227, num2:num2+227]
			cv2.imwrite(outpath + "_frame_%d.jpg" % count, image)
		count += 1
		success,image = vidcap.read()

def main():
	count = 1
	for index, split in enumerate(output_list):
		filename = os.path.join(root_path, file_list[index])
		with open(filename) as f:
			for vidname in f:
				print count
				count += 1
				vidname = vidname.strip().split('/')[-1]
				if vidname in video_list:
					vidpath = os.path.join(video_folder, vidname)
					outpath = os.path.join(output_folder,split)
					saveframes(vidpath, outpath)
				else:
					print vidname


if __name__ == '__main__':
	main()
	# saveframes('/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/video/BIG_MOMMAS_LIKE_FATHER_LIKE_SON_DVS32.avi','/home1/ayushmn/thesis/mymodels/s2vt_torch/dataset/images')
