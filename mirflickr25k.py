import torch.utils.data as data
import json
import os, sys
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *
import pandas as pd
import csv

urls = {
	"image_set": "http://press.liacs.nl:8080/mirflickr/mirflickr25k.v3/mirflickr25k.zip",
	"annotation_set": "http://press.liacs.nl:8080/mirflickr/mirflickr25k.v3/mirflickr25k_annotations_v080.zip"}

# 24 classes
object_categories = ['animals', 'baby', 'bird', 'car','clouds','dog',
                     'female', 'flower', 'food','indoor','lake','male',
                     'night','people','plant_life','portrait','river','sea',
                     'sky','structures', 'sunset', 'transport','tree','water',
                     ]

# if dataset is all prepared
def get_all_annotation_txt(root_dir):
	'''
	get the all annotation txt file name
	:param root:
	:return:
	'''
	txt_name_list = []
	class_name = []
	file_name_list = []
	base_root = ''
	all_txts = []
	all_dirs = []
	for root, all_dirs, all_txts in os.walk(root_dir):
		base_root = root
		for file in all_txts:
			f_str = list(os.path.splitext(file))
			if f_str[-1] == '.txt' and 'r1' not in str(f_str[0]) and "READ" not in str(f_str[0]):
				txt_name_list.append(os.path.join(root, file))
				file_name_list.append(file)
				class_name.append(f_str[0])
	class_name = sorted(class_name)
	file_name_list = sorted(file_name_list)
	
	return txt_name_list, class_name, file_name_list, base_root


def write_csv(path_name, class_name, file_name, csv_file, root):
	'''
	transform txt into csv
	:param path_name:   txt path+name
	:param csv_destination: csv file destination
	:return:
	'''
	img_label_dict = {}
	for i in range(len(path_name)):
		# print("file_name[i]=",file_name[i])
		txt_name = str(file_name[i].split('.')[-2])
		# print("path_name[i]=",path_name[i])
		with open(str(root + "/" + file_name[i]), 'r') as f:
			all_lines = f.readlines()
			for item in all_lines:
				info = item.split()[0]
				if str(info) not in img_label_dict.keys():
					img_label_dict[str(info)] = [-1 for ii in range(len(class_name))]
				img_label_dict[str(info)][class_name.index(txt_name)] = 1
		# print(img_label_dict)
	
	# write a csv file
	print('[dataset] write file %s' % csv_file)
	with open(csv_file, 'w') as csvfile:
		fieldnames = ['name']
		fieldnames.extend(class_name)
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
		
		for (name, labels) in img_label_dict.items():
			example = {'name': name}
			for i in range(len(class_name)):
				example[fieldnames[i + 1]] = int(labels[i])
			writer.writerow(example)
	
	csvfile.close()


def download_mirflickr25k(root, phase):
	if not os.path.exists(root):
		os.makedirs(root)
	tmpdir = os.path.join(root, 'tmp/')  # .zip file here
	data = os.path.join(root, 'data/')  # unzip file here
	if not os.path.exists(data):
		os.makedirs(data)
	if not os.path.exists(tmpdir):
		os.makedirs(tmpdir)
	filename = ''
	if phase == 'image_set':
		filename = 'mirflickr25k.zip'
	elif phase == 'annotation_set':
		filename = 'mirflickr25k_annotations_v080.zip'
	cached_file = os.path.join(tmpdir, filename)
	if not os.path.exists(cached_file):
		print('Downloading: "{0}" to {1}\n'.format(urls[phase], cached_file))
		os.chdir(tmpdir)
		subprocess.call('wget ' + urls[phase], shell=True)
		# chage directory into root
		os.chdir(root)
	
	# extract image file
	img_data = os.path.join(data, filename.split('.')[0])
	if not os.path.exists(img_data):
		print('[dataset] Extracting zip file {file} to {path}'.format(file=cached_file, path=data))
		# unzip coco dataset into tmp directory
		command = 'unzip {0} -d {1}'.format(cached_file, data)
		os.system(command)
	
	if phase == 'image_set':
		print('[mirflickr25k image dataset] Done!')
	if phase == 'annotation_set':
		print('[mirflickr25k annotation dataset] Done!')


def read_object_labels_csv(file, header=True):
	images = []
	num_categories = 0
	print('[dataset] read', file)
	with open(file, 'r') as f:
		reader = csv.reader(f)
		rownum = 0
		for row in reader:
			if header and rownum == 0:
				header = row
			else:
				if num_categories == 0:
					num_categories = len(row) - 1
				name = row[0]
				labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
				labels = torch.from_numpy(labels)
				item = (name, labels)
				images.append(item)
			rownum += 1
	return images
	
def pandas_split(path,per=0.4):
	'''
	split one .csv into train.csv and test.csv, use only once, because of the samples are selected randomly
	:param path:
	:param per:
	:return:
	'''
	df = pd.read_csv(path, encoding='utf-8')
	df = df.sample(frac=1.0)
	cut_idx = int(round(per * df.shape[0]))
	df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
	df_test.to_csv('./mirflickr25k_test.csv')
	df_train.to_csv('./mirflickr25k_train.csv')
	print(df.shape, df_test.shape, df_train.shape)
	
# def generate_all_label_csv():
	# # 根据原始的mirflickr25k的标签txt生成所有的图像的对应的标签的.csv文件
	# root_dir = \
	# 	"/data/xieyanzhao_files_in_data/Hash_coder/ML_GCN_Modified_20191120/data/mirflickr_25K/data/mirflickr25k_annotations_v080"
	# path_name,class_name,file_name,base_root=get_all_annotation_txt(root_dir)
	# write_csv(path_name,class_name,file_name,'./mirflickr_img_label.csv',base_root)

class MirFlickr25kPreProcessing(data.Dataset):
	def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
		print("load {0} file\n".format(inp_name))
		self.root = root
		self.path_images = os.path.join(root, 'data','mirflickr25k')
		# self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
		self.set = set  # its 'trainval' or 'test'
		self.transform = transform
		self.target_transform = target_transform
		
		# download dataset
		download_mirflickr25k(self.root,'image_set')
		download_mirflickr25k(self.root, 'annotation_set')
		
		# define path of csv file
		path_csv = os.path.join(self.root, 'csv_files')  # /files/VOC2007
		# define filename of csv file
		file_csv = os.path.join(path_csv, 'mirflickr25k_' + set + '.csv')
		# create the csv file if necessary
		if not os.path.exists(file_csv):
			if not os.path.exists(path_csv):  # create dir if necessary
				os.makedirs(path_csv)
		
		self.classes = object_categories
		self.images = read_object_labels_csv(file_csv)
		
		with open(inp_name, 'rb') as f:
			self.inp = pickle.load(f)
		self.inp_name = inp_name
		
		print('[dataset] MirFlickr25k classification set=%s number of classes=%d  number of images=%d' % (
			set, len(self.classes), len(self.images)))
	
	def __getitem__(self, index):
		
		path, target = self.images[index]
		# print("MirFlickr25kPreprocessing __getitem__ says:path=\n{0},target=\n{1}".format(path, target))
		# print("self.path_iamges=",self.path_images)
		img = Image.open(os.path.join(self.path_images,'im'+ path + '.jpg')).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return (img, path, self.inp), target
	
	def __len__(self):
		'''
		return the amount of elements
		:return:
		'''
		return len(self.images)
	
	def get_number_classes(self):
		return len(self.classes)
	
	@property
	def display_info(self):
		print(
			"self.root={0},\nself.path_images={1},\
			\nself.set={2},\nself.transform={3},\nself.target_transform={4}\n" \
				.format(self.root, self.path_images, self.set,
			            self.transform, self.target_transform))


if __name__ == "__main__":
	path = './data/mirflickr_img_label.csv'
	# obj = MirFlickr25kPreProcessing('./data/mirflickr25k/', 'train',
	#                                       inp_name='data/mirflickr25k/mirflickr25k_glove_word2vec.pkl')
	#
	# obj = MirFlickr25kPreProcessing('./data/mirflickr25k/', 'test',
	#                                       inp_name='data/mirflickr25k/mirflickr25k_glove_word2vec.pkl')
	
	pandas_split(path, 0.5)
	