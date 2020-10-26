#################################################################
# this file includes the method for processing pkl file
# indentify the interaction set between target_gt labels
# calc the dist between the output labels
# there is only one file neeed process, hashcode_pool.pkl
#################################################################

import pickle, torch, os, sys, math, datetime
from tqdm import tqdm, tgrange
import numpy as np


class PostPro(object):
	def __init__(self, state={}):
		super(PostPro, self).__init__()
		self.state = state
		self.all_cheat_item = 0
		
		if self._state('use_gpu') is None:
			self.state['use_gpu'] = torch.cuda.is_available()
		# self.state['use_gpu'] = False
		
		if self._state('prefix_path') is None:
			tmp = self.state['hashcode_pool'].split('/')[:-1]
			self.state['prefix_path'] = '/'.join(tmp) + '/'
		# print(self.state['prefix_path'])
		
		if self._state('cheat_hashcode_pool') is None:
			self.state['cheat_hashcode_pool'] = self.state['prefix_path'] + \
												"cheat_" + self.state['hashcode_pool'].split('/')[-1]
		
		if self._state('before_fc_destination') is None:
			self.state['before_fc_destination'] = ''
		
		if self._state('hashcode_pool_image_name_list') is None:
			self.state['hashcode_pool_image_name_list'] = []
		
		if self._state('cheat_hashcode_pool_image_name_list') is None:
			self.state['cheat_hashcode_pool_image_name_list'] = []
		
		if self._state('query_pool_image_name_list') is None:
			self.state['query_pool_image_name_list'] = []
		
		# if self._state('hash_bit') is None:
		# 	self.state['hash_bit'] = 64
		if self._state('threshold') is None:
			self.state['threshold'] = 0.09
		
		if self._state('redius') is None:
			self.state['redius'] = 2
		
		if self._state('hashcode_pool_limit') is None:
			self.state['hashcode_pool_limit'] = 3000
		# voc 3000
		# mirflickr25k 6000
		# ms_coco 25000
		
		if self._state('query_pool_limit') is None:
			self.state['query_pool_limit'] = 100
		
		self.init_variables()
	
	def _state(self, name):
		if name in self.state:
			return self.state[name]
	
	@staticmethod
	def calc_mean_var(nlist):
		"""
		calculate the mean and var of a list which named "nlist"
		:param nlist:
		:return:
		"""
		N = float(len(nlist))
		narray = np.array(nlist)
		sum1 = float(narray.sum())
		narray2 = narray * narray
		sum2 = float(narray2.sum())
		mean = sum1 / N
		var = sum2 / N - mean ** 2  # D(X) = E(X^2) - E(X)^2
		
		return mean, var
	
	def _deleterow(self, ts, idx):
		'''
		delete a row in tensor
		:param idx:
		:return:
		'''
		ts = ts[torch.arange(ts.size(0)) != idx]
		print(ts)
	
	def reset(self):
		"""Resets the meter with empty member variables"""
		# self.output will store all the dataset(train_set or test set) output info
		if self.state['use_gpu']:
			self.output = torch.IntTensor(torch.IntStorage()).cuda()
			# #print("In the class AveragePrecisionMeter function reset(): self.score.shape=,self.score=,self.score.type",
			#       self.output.shape, "\n",self.output,"\n",type(self.output))
			# self.output will store all the dataset(train_set or test set) labels info
			self.targets = torch.IntTensor(torch.IntStorage()).cuda()
		else:
			self.output = torch.IntTensor(torch.IntStorage())
			self.targets = torch.IntTensor(torch.IntStorage())
		
		## stack all the names of every img
		self.img_name = []
		
		if self.state['use_gpu']:
			self.bf_output = torch.FloatTensor(torch.FloatStorage()).cuda()
			self.bf_targets = torch.IntTensor(torch.IntStorage()).cuda()
		else:
			self.bf_output = torch.FloatTensor(torch.FloatStorage())
			self.bf_targets = torch.IntTensor(torch.IntStorage())
		self.all_item = 0
		self.all_cheat_item = 0
		self.mAP1_list = []
		self.mAP2_list = []
		self.P_list = []
		self.R_list = []
	
	def init_variables(self):
		self.state['query_pool_image_name_list'] = []
		self.state['final_hashcode_pool_image_name_list'] = []
		self.state['hashcode_pool_image_name_list'] = []
		if type(self.state['start_time']) == type('123'):
			self.state['hashcode_pool'] = os.path.splitext(self.state['hashcode_pool'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'] + '.pkl'
			self.state['query_pool'] = os.path.splitext(self.state['query_pkl_path'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'] + '.pkl'
		else:
			self.state['hashcode_pool'] = os.path.splitext(self.state['hashcode_pool'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'].strftime('%Y%m%d%H%M%S') + '.pkl'
			self.state['query_pool'] = os.path.splitext(self.state['query_pkl_path'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'].strftime('%Y%m%d%H%M%S') + '.pkl'
		self.state['final_hashcode_pool'] = self.state['prefix_path'] + \
											"final_" + self.state['hashcode_pool'].split('/')[-1]
		self.state['before_fc_destination'] = self.state['hashcode_pool'][:-4] + 'before_fc.pkl'
	
	def _gpu_cpu(self, input):
		'''
		adapt gpu or cpu environment
		:param input:
		:return:
		'''
		if torch.is_tensor(input):
			if self.state['use_gpu']:
				input = input.float()  # cuda only support float type tensor
				return input.cuda()
		return input.float()
	
	def _wbin_pkl(self, file_dir, content):
		'''write content into .pkl file with bin format'''
		with open(str(file_dir), 'ab') as fi:
			pickle.dump(content, fi)
	
	def stack_from_src(self, ):
		print('search the file named {0}\n'.format(self.state['hashcode_pool']))
		if os.path.exists(self.state['hashcode_pool']):
			print("The {0} file exists...\n".format(self.state['hashcode_pool']))
			u = 0
			with open(self.state['hashcode_pool'], 'rb') as f:
				while True:
					u += 1  # u = u+1
					try:
						data = pickle.load(f)
						# print("data=\n{0},\n{1}".format(data, type(data)))
						name = data['img_name']
						target_gt = data['target'].cpu().int()
						output = data['output'].cpu().int()
						self.state['hashcode_pool_image_name_list'].extend(name)
						if self.state['use_gpu']:
							self.output = torch.cat((self.output, output.reshape(output.size(0), -1).cuda()), 0)  #
							self.targets = torch.cat((self.targets, target_gt.reshape(target_gt.size(0), -1).cuda()), 0)
						else:
							self.output = torch.cat((self.output, output.reshape(output.size(0), -1)), 0)  #
							self.targets = torch.cat((self.targets, target_gt.reshape(target_gt.size(0), -1)), 0)
						self.img_name.extend(name)
						self.all_item += len(name)
					except EOFError:
						break
		
		else:
			print("Cannot find the {0} file ! Please check...\n The process aborting... ... ! \n".
				  format(self.state['hashcode_pool']))
			sys.exit()
	
	# print("self.target = \n{0}, \nself.output = \n{1} ". \
	#       format(self.targets, self.output))
	# print("self.target.shape = \n{0}, \nself.output.shape = \n{1} ". \
	#       format(self.targets.shape, self.output.shape))
	
	def createSeveralMat(self, query_num):
		# select a query hash code
		query_code = self.output[query_num, :]
		# corresponding gt-labels
		query_label = self.targets[query_num, :]
		
		# calc the dist vector
		# print("query code is:\n",query_code)
		tmp = (torch.mul(query_code, self.output) + 1) / 2  # 减一除以(-2)
		dist_mat = self.state['hash_bit'] - torch.sum(tmp, 1)
		# print("tmp is:\n{0} \n{1}".format(dist_mat, dist_mat.shape))
		
		# calc the interact set betweent different targets
		tmp = torch.matmul(self.targets.float(), torch.transpose(self.targets.float(), 0, 1))
		bak = tmp[query_num, :].reshape(1, -1)
		# print("bak=",'\n',bak)
		# print()
		all_gt_match_num = torch.nonzero(bak).size(0)
		# print("all_gt_match_num=", '\n', all_gt_match_num)
		# print()
		upper_tri = torch.triu(tmp, diagonal=1)
		upper_tri += torch.transpose(upper_tri, 0, 1)  # symmetric matrix , diagonal elements are 0
		# print("\ninteract set mat is:\n{0}, \n{1}".format(upper_tri, upper_tri.shape))
		
		mAP = 0
		seq = 0  # seq num
		shot = 0  # shot target
		redius_pool = {}
		interset_pool = {}
		for i in range(dist_mat.size(0)):
			if i != query_num:
				if dist_mat[i] <= int(self.state['redius']):
					if int(dist_mat[i]) not in redius_pool:
						redius_pool[int(dist_mat[i])] = {}
					redius_pool[int(dist_mat[i])][int(i)] = int(upper_tri[query_num][i])
		all_keys = []
		for k, v in redius_pool.items():
			all_keys.append(k)
			v = sorted(v.items(), key=lambda item: item[1], reverse=True)
			redius_pool[k] = v
		# the `key` means the hamming distance
		all_keys.sort()
		# print("all_keys={0}\n".format(all_keys))
		# print()
		# print("\nbefore sorted in terms of key:redius_pool=\n{0}".format(redius_pool))
		# print()
		
		num_within_redius = 0
		
		for ele in all_keys:
			vl = redius_pool[int(ele)]
			# the num of vectors within the hanmming redius
			num_within_redius += len(vl)
			for i in range(len(vl)):
				seq += 1
				t_e = vl[i][1]  # the vl maybe 2, 1, 0
				if t_e:
					shot += 1
					mAP += shot / seq
		
		mAP1, mAP2 = mAP, mAP
		P = shot / num_within_redius if num_within_redius else None
		R = shot / all_gt_match_num if all_gt_match_num else None
		mAP1 = (mAP / shot) if shot else None  # map / shot
		mAP2 = (mAP / seq) if seq else None  # map / seq
		print("img name is :", self.img_name[query_num])
		print("hashcode:\n", query_code)
		print("shot={0},all_gt_match_num={1}".format(shot, all_gt_match_num))
		print("mAP/shot vlaue = {0},\nmAP/seq value = {1},\nPrecision = {2},\nRecall = {3}". \
			  format(mAP1, mAP2, P, R))
		# store the value in the member variables
		if mAP1:
			self.mAP1_list.append(mAP1)
		if mAP2:
			self.mAP2_list.append(mAP2)
			self.P_list.append(P)
			self.R_list.append(R)
		print()
		
		if mAP2 != None and mAP2 > self.state['threshold']:
			cheat_dic = {  # 'id': self.state['hashcode_pool_image_name_list'][query_num],
				'gt': self.targets[query_num, :].char(),
				'out': self.output[query_num, :].char(),
			}
			self.state['cheat_hashcode_pool_image_name_list'].append(
				(self.state['hashcode_pool_image_name_list'][query_num], mAP2, P, R, cheat_dic)
			)
		
		if mAP2 != None and mAP2 > 0.7:
			query_dic = {  # 'id': self.state['hashcode_pool_image_name_list'][query_num],
				'gt': self.targets[query_num, :].char(),
				'out': self.output[query_num, :].char(),
			}
			self.state['query_pool_image_name_list'].append(
				(self.state['hashcode_pool_image_name_list'][query_num], mAP2, P, R, query_dic)
			)
	
	def create_cheat_pkl(self):
		print('create_cheat_pkl says:\n')
		
		self.state['cheat_hashcode_pool_image_name_list'].sort(key=lambda x: x[1], reverse=True)
		self.state['query_pool_image_name_list'].sort(key=lambda x: x[1], reverse=True)
		
		# 截断
		self.state['cheat_hashcode_pool_image_name_list'] = \
			self.state['cheat_hashcode_pool_image_name_list'][:self.state['hashcode_pool_limit']]
		self.state['query_pool_image_name_list'] = \
			self.state['query_pool_image_name_list'][:self.state['query_pool_limit']]
		
		print("create_cheat_pkl says:\n")
		for i in range(len(self.state['cheat_hashcode_pool_image_name_list'])):
			content = self.state['cheat_hashcode_pool_image_name_list'][i]
			# use cpu data to store
			tmp_dic = {'id': content[0],
					   'target_01': content[-1]['gt'].cpu().char(),
					   'output': content[-1]['out'].cpu().char(),
					   }
			# print("tmp_dic=\n{0},".format(tmp_dic))
			self._wbin_pkl(self.state['cheat_hashcode_pool'], tmp_dic)
		print("cheat_pool content output over...\n\n")
		print('print query_pool content\n')
		for i in range(len(self.state['query_pool_image_name_list'])):
			content = self.state['query_pool_image_name_list'][i]
			# use cpu data to store
			tmp_dic = {'id': content[0],
					   'target_01': content[-1]['gt'].cpu().char(),
					   'output': content[-1]['out'].cpu().char(),
					   }
			# print("tmp_dic=\n{0},".format(tmp_dic))
			self._wbin_pkl(self.state['query_pool'], tmp_dic)
	
	def cheat_stack_tensor(self, ):
		print("cheat_stack_tensor says:\n")
		print("processing the file named {0}".format(self.state['cheat_hashcode_pool']))
		if os.path.exists(self.state['cheat_hashcode_pool']):
			u = 0
			with open(self.state['cheat_hashcode_pool'], 'rb') as f:
				while True:
					try:
						data = pickle.load(f)
						# print("data=\n{0},\n{1}".format(data, type(data)))
						name = data['id']
						target_01 = data['target_01'].cpu().int()
						output = data['output'].cpu().int()
						if self.state['use_gpu']:
							self.output = torch.cat((self.output, output.reshape(1, -1).cuda()), 0)  #
							self.targets = torch.cat((self.targets, target_01.reshape(1, -1).cuda()), 0)
						else:
							self.output = torch.cat((self.output, output.reshape(1, -1)), 0)  #
							self.targets = torch.cat((self.targets, target_01.reshape(1, -1)), 0)
						self.img_name.append(name)
					except EOFError:
						break
					u += 1  # u = u+1
			self.all_cheat_item = u
			print("item count = ", self.all_cheat_item)
		else:
			print("cannot find the file named {0}\n Processing aborting... ...\n".
				  format(self.state['cheat_hashcode_pool']))
			sys.exit()
	
	# print("self.target = \n{0}, \nself.output = \n{1} ". \
	#       format(self.targets, self.output))
	# print("self.target.shape = \n{0}, \nself.output.shape = \n{1} ". \
	#       format(self.targets.shape, self.output.shape))
	
	def cheat_calc(self, query_num):
		# select a query hash code
		query_code = self.output[query_num, :]
		# corresponding gt-labels
		query_label = self.targets[query_num, :]
		
		# calc the dist vector
		# print("query code is:\n",query_code)
		tmp = (torch.mul(query_code, self.output) + 1) / 2  # 减一除以(-2)
		dist_mat = self.state['hash_bit'] - torch.sum(tmp, 1)
		# print("tmp is:\n{0} \n{1}".format(dist_mat, dist_mat.shape))
		
		# calc the interact set betweent different targets
		tmp = torch.matmul(self.targets.float(), torch.transpose(self.targets.float(), 0, 1))
		bak = tmp[query_num, :].reshape(1, -1)
		# print("bak=",'\n',bak)
		# print()
		all_gt_match_num = torch.nonzero(bak).size(0)  # 这里是所有与查询的hashcode的gt标签有交集的数量
		# print("all_gt_match_num=", '\n', all_gt_match_num)
		# print()
		upper_tri = torch.triu(tmp, diagonal=1)
		upper_tri += torch.transpose(upper_tri, 0, 1)  # symmetric matrix , diagonal elements are 0
		# print("\ninteract set mat is:\n{0}, \n{1}".format(upper_tri, upper_tri.shape))
		
		mAP = 0
		seq = 0  # seq num
		shot = 0  # shot target
		redius_pool = {}
		interset_pool = {}
		for i in range(dist_mat.size(0)):
			if i != query_num:
				if dist_mat[i] <= int(self.state['redius']):
					if int(dist_mat[i]) not in redius_pool:
						redius_pool[int(dist_mat[i])] = {}
					redius_pool[int(dist_mat[i])][int(i)] = int(upper_tri[query_num][i])
		all_keys = []
		for k, v in redius_pool.items():
			all_keys.append(k)
			v = sorted(v.items(), key=lambda item: item[1], reverse=True)
			redius_pool[k] = v
		all_keys.sort()  # re-ranking the query result
		# print("all_keys={0}\n".format(all_keys))
		# print()
		# print("\nbefore sorted in terms of key:redius_pool=\n{0}".format(redius_pool))
		# print()
		
		num_within_redius = 0
		
		for ele in all_keys:
			vl = redius_pool[int(ele)]
			# the num of vectors within the hanmming redius
			num_within_redius += len(vl)
			for i in range(len(vl)):
				seq += 1
				t_e = vl[i][1]
				if t_e:
					shot += 1
					mAP += shot / seq
		
		# mAP1, mAP2 = mAP, mAP
		P = shot / num_within_redius if num_within_redius else None
		R = shot / all_gt_match_num if all_gt_match_num else None
		mAP1 = (mAP / shot) if shot else None  # map / shot
		mAP2 = (mAP / seq) if seq else None  # map / seq
		print("img name is :", self.img_name[query_num])
		print("hashcode:\n", query_code)
		print("shot={0},all_gt_match_num={1}".format(shot, all_gt_match_num))
		print("mAP/shot vlaue = {0},\nmAP/seq value = {1},\nPrecision = {2},\nRecall = {3}". \
			  format(mAP1, mAP2, P, R))
		if mAP1:
			self.mAP1_list.append(mAP1)
		if mAP2:
			self.mAP2_list.append(mAP2)
			self.P_list.append(P)
			self.R_list.append(R)
		print()
	
	def read_format(self):
		print('self.state["before_fc_destination"] = ', self.state['before_fc_destination'])
		if os.path.exists(self.state['before_fc_destination']):
			u = 0
			with open(self.state['before_fc_destination'], 'rb') as f:
				while True:
					u += 1  # u = u+1
					try:
						data = pickle.load(f)
						# print("data=\n{0},\n{1}".format(data, type(data)))
						name = data['img_name']
						target_gt = data['target'].cpu().int()
						output = data['output'].cpu().float()
						# self.state['hashcode_pool_image_name_list'].extend(name)
						if self.state['use_gpu']:
							self.bf_output = torch.cat((self.bf_output, output.reshape(output.size(0), -1).cuda()),
													   0)  #
							self.bf_targets = torch.cat(
								(self.bf_targets, target_gt.reshape(target_gt.size(0), -1).cuda()), 0)
						else:
							self.bf_output = torch.cat((self.bf_output, output.reshape(output.size(0), -1)), 0)  #
							self.bf_targets = torch.cat((self.bf_targets, target_gt.reshape(target_gt.size(0), -1)), 0)
						self.all_item += len(name)
					except EOFError:
						break
			print('self.bf_output shape={0}, \ncontent=\n{1}'.format(self.bf_output.shape, self.bf_output))
			print('self.bf_target shape={0}, \ncontent=\n{1}'.format(self.bf_targets.shape, self.bf_targets))
	
	def test_cheat(self):
		print('Start test cheating hash pool...')
		print()
		self.reset()
		# self.display
		self.cheat_stack_tensor()
		if self.all_cheat_item:
			iteration = tqdm(range(self.all_cheat_item), desc='CheatTest')
			for i in iteration:
				self.cheat_calc(i)
			print()
			print("*" * 10, 'Cheat overall mean mAP1={0}'.format(self.calc_mean_var(self.mAP1_list)[0]), '*' * 10, '\n')
			print("*" * 10, 'Cheat overall mean mAP2={0}'.format(self.calc_mean_var(self.mAP2_list)[0]), '*' * 10, '\n')
			print("*" * 10, 'Cheat overall mean P={0}'.format(self.calc_mean_var(self.P_list)[0]), '*' * 10, '\n')
			print("*" * 10, 'Cheat overall mean R={0}'.format(self.calc_mean_var(self.R_list)[0]), '*' * 10, '\n')
		else:
			print('Cheat pool have no value')
	
	def select_img(self):
		print('Start select hash code...')
		print()
		self.reset()
		# self.display
		self.stack_from_src()
		if os.path.exists(self.state['cheat_hashcode_pool']):
			os.remove(self.state['cheat_hashcode_pool'])
		if os.path.exists(self.state['query_pool']):
			os.remove(self.state['query_pool'])
		# voc test set includes 4592 imgs
		# coco test set includes 40137 imgs
		iteration = tqdm(range(self.all_item), desc='OriginTest')
		for i in iteration:
			# self.createSeveralMat(i)
			self.createSeveralMat(i)
		print()
		print("*" * 10, 'Original overall mean mAP1={0}'.format(self.calc_mean_var(self.mAP1_list)[0]), '*' * 10, '\n')
		print("*" * 10, 'Original overall mean mAP2={0}'.format(self.calc_mean_var(self.mAP2_list)[0]), '*' * 10, '\n')
		print("*" * 10, 'Original overall mean P={0}'.format(self.calc_mean_var(self.P_list)[0]), '*' * 10, '\n')
		print("*" * 10, 'Original overall mean R={0}'.format(self.calc_mean_var(self.R_list)[0]), '*' * 10, '\n')
		self.create_cheat_pkl()
	
	@property
	def display(self, ):
		for k, v in self.state.items():
			print("{0}={1}".format(k, v))
		print("all_cheat_item={0}".format(self.all_cheat_item))
		print("all_item={0}".format(self.all_item))
	
	def read_bf_info(self):
		self.reset()
		self.read_format()


if __name__ == "__main__":
	state_voc = {'num_classes': 20,
				 # 'testset_pkl_path': './data/voc/voc_test_set.pkl',
				 'hashcode_pool': './data/voc/voc_hashcode_pool.pkl',
				 'query_pkl_path': './data/voc/voc_query_set.pkl',
				 'start_time': str(201912232152),
				 'hash_bit': 64,
				 'hashcode_pool_limit': 3000,
				 }
	state_mirflickr25k = {'num_classes': 24,
						  # 'testset_pkl_path': './data/mirflickr25k/mirflickr25k_test_set.pkl',
						  'hashcode_pool': './data/mirflickr25k/mirflickr25k_hashcode_pool.pkl',
						  'query_pkl_path': './data/mirflickr25k/mirflickr25k_query_set.pkl',
						  'start_time': str(201912251332),
						  'hash_bit': 64,
						  'hashcode_pool_limit': 6000,
						  }
	state_voc1 = {'num_classes': 20,
				  # 'testset_pkl_path': './data/voc/voc_test_set.pkl',
				  'hashcode_pool': './data/voc/voc_hashcode_pool.pkl',
				  'query_pkl_path': './data/voc/voc_query_set.pkl',
				  'start_time': str(202001032023),
				  'hash_bit': 128,
				  'hashcode_pool_limit': 3000,
				  }
	state_coco1 = {'num_classes': 80,
				   # 'testset_pkl_path': './data/coco/coco_test_set.pkl',
				   'hashcode_pool': './data/coco/coco_hashcode_pool.pkl',
				   'query_pkl_path': './data/coco/coco_query_set.pkl',
				   'start_time': str(202001081640),
				   'hash_bit': 16,
				   'hashcode_pool_limit': 25000,
				   }
	state_voc2 = {'num_classes': 20,
				  # 'testset_pkl_path': './data/voc/voc_test_set.pkl',
				  'hashcode_pool': './data/voc/voc_hashcode_pool.pkl',
				  'query_pkl_path': './data/voc/voc_query_set.pkl',
				  'start_time': str(202001092328),
				  'hash_bit': 128,
				  'hashcode_pool_limit': 3000,
				  }
	obj = PostPro(state_voc2)
	obj.select_img()
	obj.test_cheat()
# obj.read_bf_info()
