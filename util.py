import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import numpy as np
import torch.nn.functional as F
import Global_Loss
from cauchy_hash import *

# DEBUG switch
DEBUG_UTIL = False


class Warp(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		# if DEBUG_UTIL:
		#     print("FILE:util.py CLASS:Warp INIT\n")
		
		self.size = int(size)
		self.interpolation = interpolation
	
	def __call__(self, img):
		# if DEBUG_UTIL:
		#     print("FILE:util.py CLASS:Warp FUNC:__call__\n")
		
		return img.resize((self.size, self.size), self.interpolation)
	
	def __str__(self):
		# if DEBUG_UTIL:
		#     print("FILE:util.py CLASS:Warp FUNC:__str__\n")
		return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
																								interpolation=self.interpolation)


class MultiScaleCrop(object):
	'''
	Get many images which have different scale
	'''
	
	def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
		
		self.scales = scales if scales is not None else [1, 875, .75, .66]
		self.max_distort = max_distort
		self.fix_crop = fix_crop
		self.more_fix_crop = more_fix_crop
		self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
		self.interpolation = Image.BILINEAR  # bilinear interpolation (双线性插值)
	
	def __call__(self, img):
		
		im_size = img.size
		crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
		crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
		ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
		return ret_img_group
	
	def _sample_crop_size(self, im_size):
		image_w, image_h = im_size[0], im_size[1]
		
		# find a crop size
		base_size = min(image_w, image_h)
		crop_sizes = [int(base_size * x) for x in self.scales]
		crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
		crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
		
		pairs = []
		for i, h in enumerate(crop_h):
			for j, w in enumerate(crop_w):
				if abs(i - j) <= self.max_distort:
					pairs.append((w, h))
		
		crop_pair = random.choice(pairs)
		if not self.fix_crop:
			w_offset = random.randint(0, image_w - crop_pair[0])
			h_offset = random.randint(0, image_h - crop_pair[1])
		else:
			w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
		
		return crop_pair[0], crop_pair[1], w_offset, h_offset
	
	def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
		offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
		return random.choice(offsets)
	
	@staticmethod
	def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
		
		w_step = (image_w - crop_w) // 4
		h_step = (image_h - crop_h) // 4
		
		ret = list()
		ret.append((0, 0))  # upper left
		ret.append((4 * w_step, 0))  # upper right
		ret.append((0, 4 * h_step))  # lower left
		ret.append((4 * w_step, 4 * h_step))  # lower right
		ret.append((2 * w_step, 2 * h_step))  # center
		
		if more_fix_crop:
			ret.append((0, 2 * h_step))  # center left
			ret.append((4 * w_step, 2 * h_step))  # center right
			ret.append((2 * w_step, 4 * h_step))  # lower center
			ret.append((2 * w_step, 0 * h_step))  # upper center
			
			ret.append((1 * w_step, 1 * h_step))  # upper left quarter
			ret.append((3 * w_step, 1 * h_step))  # upper right quarter
			ret.append((1 * w_step, 3 * h_step))  # lower left quarter
			ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
		
		return ret
	
	def __str__(self):
		# if DEBUG_UTIL:
		#     print("FILE:util.py CLASS:MultiScaleCrop FUNC:__str__")
		return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
	"""Download a URL to a local file.

	Parameters
	----------
	url : str
		The URL to download.
	destination : str, None
		The destination of the file. If None is given the file is saved to a temporary directory.
	progress_bar : bool
		Whether to show a command-line progress bar while downloading.

	Returns
	-------
	filename : str
		The location of the downloaded file.

	Notes
	-----
	Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
	"""
	
	def my_hook(t):
		last_b = [0]
		
		def inner(b=1, bsize=1, tsize=None):
			if tsize is not None:
				t.total = tsize
			if b > 0:
				t.update((b - last_b[0]) * bsize)
			last_b[0] = b
		
		return inner
	
	if progress_bar:
		with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
			filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
	else:
		filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
	"""
	The APMeter measures the average precision per class.
	The APMeter is designed to operate on `NxK` Tensors `output` and
	`target`, and optionally a `Nx1` Tensor weight where (1) the `output`
	contains model output scores for `N` examples and `K` classes that ought to
	be higher when the model is more convinced that the example should be
	positively labeled, and smaller when the model believes the example should
	be negatively labeled (for instance, the output of a sigmoid function); (2)
	the `target` contains only values 0 (for negative examples) and 1
	(for positive examples); and (3) the `weight` ( > 0) represents weight for
	each sample.
	"""
	
	def __init__(self, difficult_examples=False):
		super(AveragePrecisionMeter, self).__init__()
		self.reset()
		self.difficult_examples = difficult_examples
		# print("Class AveragePrecisionMeter initiates over...")
	
	def reset(self):
		"""Resets the meter with empty member variables"""
		self.scores = torch.FloatTensor(torch.FloatStorage())
		# print("In the class AveragePrecisionMeter function reset(): self.score.shape=,self.score=,self.score.type",
		#       self.scores.shape, "\n",self.scores,"\n",type(self.scores))
		self.targets = torch.LongTensor(torch.LongStorage())
	
	def add(self, output, target):
		"""
		Args:
			output (Tensor): NxK tensor that for each of the N examples
				indicates the probability of the example belonging to each of
				the K classes, according to the model. The probabilities should
				sum to one over all classes
			target (Tensor): binary NxK tensor that encodes which of the K
				classes are associated with the N-th input
					(eg: a row [0, 1, 0, 1] indicates that the example is
						 associated with classes 2 and 4)
			weight (optional, Tensor): Nx1 tensor representing the weight for
				each example (each weight > 0)
		"""
		
		if not torch.is_tensor(output):
			output = torch.from_numpy(output)
		if not torch.is_tensor(target):
			target = torch.from_numpy(target)
		
		if output.dim() == 1:
			output = output.view(-1, 1)
		else:
			assert output.dim() == 2, \
				'wrong output size (should be 1D or 2D with one column \
				per class)'
		if target.dim() == 1:
			target = target.view(-1, 1)
		else:
			assert target.dim() == 2, \
				'wrong target size (should be 1D or 2D with one column \
				per class)'
		if self.scores.numel() > 0:
			assert target.size(1) == self.targets.size(1), \
				'dimensions for output should match previously added examples.'
		
		# make sure storage is of sufficient size
		if self.scores.storage().size() < self.scores.numel() + output.numel():
			new_size = math.ceil(self.scores.storage().size() * 1.5)
			self.scores.storage().resize_(int(new_size + output.numel()))
			self.targets.storage().resize_(int(new_size + output.numel()))
		
		# store scores and targets
		offset = self.scores.size(0) if self.scores.dim() > 0 else 0
		self.scores.resize_(offset + output.size(0), output.size(1))
		self.targets.resize_(offset + target.size(0), target.size(1))
		self.scores.narrow(0, offset, output.size(0)).copy_(output)
		self.targets.narrow(0, offset, target.size(0)).copy_(target)
	
	def value(self):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		# print("(util-1)Go into the value func of class AveragePrecisionMeter\n")
		if self.scores.numel() == 0:
			return 0
		ap = torch.zeros(self.scores.size(1))
		# print("(util-2)ap = ",ap,"ap.shape = ",ap.shape,'\n')
		rg = torch.arange(1, self.scores.size(0)).float()
		# print("(util-3)class AveragePrecisionMeter func value:\n size of self.scores.size ,self.scores ",self.scores.size(),"\n",self.scores)
		# print("(util-4)class AveragePrecisionMeter func value in for loop: "
		#       "size of self.scores = \n", self.scores.size(), "\n", self.scores, "\n")
		# print("(util-5)class AveragePrecisionMeter func value in for loop: "
		#       "size of self.targets = \n", self.targets.size(), "\n", self.targets, "\n")
		# self.scores = self.scores[0:int(self.targets.size(0)), :]
		# compute average precision for each class
		for k in range(self.scores.size(1)):  # k from 0 to 19
			# sort scores
			# k from 0 to 20
			scores = self.scores[:, k]
			targets = self.targets[:, k]
			# compute average precision
			ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
		return ap
	
	@staticmethod
	def average_precision(output, target, difficult_examples=True):
		# sort examples
		sorted, indices = torch.sort(output, dim=0, descending=True)
		# print("(util-6)util.py file average_precision func:\n","type(sorted) = ",type(sorted),
		#       "\nsorted.shape = ",sorted.shape,"\ntype(indices) = ",type(indices),"\nindices.shape = ",indices.shape)
		
		# Computes prec@i
		pos_count = 0.
		total_count = 0.
		precision_at_i = 0.
		# print("(util-7)util.py file average_precision func: type(indices), indices",type(indices),'\n', indices,"\n")
		
		for i in indices:
			# print("(util-8)util.py file average_precision func: i, type(i)",i,type(i),"\n")
			label = target[i]
			# print("(util-9)util.py file average_precision func: label, type(label)", label, type(label), "\n")
			if difficult_examples and label == 0:
				continue
			if label == 1:
				pos_count += 1
			total_count += 1
			if label == 1:
				precision_at_i += pos_count / total_count
		precision_at_i /= pos_count
		return precision_at_i
	
	def overall(self):
		if self.scores.numel() == 0:
			return 0
		scores = self.scores.cpu().numpy()
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		return self.evaluation(scores, targets)
	
	def overall_topk(self, k):
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		n, c = self.scores.size()
		scores = np.zeros((n, c)) - 1
		index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
		tmp = self.scores.cpu().numpy()
		for i in range(n):
			for ind in index[i]:
				scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
		return self.evaluation(scores, targets)
	
	def evaluation(self, scores_, targets_):
		n, n_class = scores_.shape
		Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
		for k in range(n_class):
			scores = scores_[:, k]
			targets = targets_[:, k]
			targets[targets == -1] = 0
			Ng[k] = np.sum(targets == 1)
			Np[k] = np.sum(scores >= 0)
			Nc[k] = np.sum(targets * (scores >= 0))
		Np[Np == 0] = 1
		# print("Nc={0},\n{1}, Np={2},\n{3}, Ng={4},\n{5}".format(Nc,Nc.shape,Np,Np.shape,Ng,Ng.shape))
		OP = np.sum(Nc) / np.sum(Np)
		OR = np.sum(Nc) / np.sum(Ng)
		OF1 = (2 * OP * OR) / (OP + OR)
		
		CP = np.sum(Nc / Np) / n_class
		# Ng[ Ng==0 ] = 1
		CR = np.sum(Nc / Ng) / n_class
		CF1 = (2 * CP * CR) / (CP + CR)
		return OP, OR, OF1, CP, CR, CF1


class HashAveragePrecisionMeter(AveragePrecisionMeter):
	def __init__(self, difficult_examples=False):
		AveragePrecisionMeter.__init__(self, difficult_examples)
	
	def loss_value(self):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		if self.scores.numel() == 0:
			return 0
		target_gt = self.targets
		all_output = self.scores
		
		target_gt[target_gt >= 0] = 1
		target_gt[target_gt < 0] = 0
		calcloss = CauchyLoss(sij_type='IOU', normed=True)
		epoch_loss = calcloss.forward(target_gt, all_output)
		
		return epoch_loss
	
	def batch_loss_value(self, batch_target, batch_output):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		target_gt = batch_target
		all_output = batch_output
		
		target_gt[target_gt >= 0] = 1
		target_gt[target_gt < 0] = 0
		calcloss = CauchyLoss(sij_type='IOU', normed=True)
		batch_loss = calcloss.forward(target_gt, all_output)
		
		# print("HashAveragePrecisionMeter batch_loss_value:batch_loss=",batch_loss)
		# sys.exit()
		return batch_loss


def gen_A(p, num_classes, t, adj_file):
	'''
	generate the adjecent matrix
	:param opt: get command parameters
	:param num_classes: the amount of classes
	:param t:
	:param adj_file:    word embeding matrix???
	:return:
	'''
	import pickle
	# print("util.py says:t={0},p={1}".format(t, p))
	result = pickle.load(open(adj_file, 'rb'))
	_adj = result['adj']
	print('_adj = \n{0}\n'.format(_adj))
	_nums = result['nums']  # the staticstic value of images number of every class
	print('_nums = \n{0}\n'.format(_nums))
	_nums = _nums[:, np.newaxis]  # increase a dimention, transform row vector into column vector
	print('_nums = \n{0}\n'.format(_nums))
	_adj = _adj / _nums
	_adj[_adj < t] = 0  # this t is the threshold 'tao' in the formula (7) of ML_GCN
	_adj[_adj >= t] = 1
	# _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)  # 0.25 denotes the p value in the formula (8)
	_adj = _adj * p / (_adj.sum(0, keepdims=True) + 1e-6)  # keepdim means the output is 2-dimension, e.g. 1*20
	_adj = _adj + np.identity(num_classes, np.int)  # add an indentity matrix
	print("_adj.shape = ", _adj.shape)  # 20 * 20
	return _adj


def gen_adj(A):
	D = torch.pow(A.sum(1).float(), -0.5)  # sum element according to the dimension 1
	print("D.shape is :", D.shape, "\n", D)
	D = torch.diag(D)  # diagnal elements
	print("D.shape is :", D.shape, "\n", D)
	adj = torch.matmul(torch.matmul(A, D).t(), D)  # (AD)_T * D = D_T * A_T * D
	print("adj.shape is :", adj.shape, "\n", adj)
	return adj


def adj2tensor(adj_file):
	import pickle
	result = pickle.load(open(adj_file, 'rb'))
	_adj = result['adj']
	_nums = result['nums']
	return _adj


def gen_correlation(A):
	I_c = torch.eye(A.shape[0])
	A_wave = A + I_c
	tmp = A_wave.sum(1).float()
	D_wave_negative_power = torch.diag(tmp ** (-0.5))
	A_hat = torch.matmul(torch.matmul(D_wave_negative_power, A_wave), D_wave_negative_power)
	# print("\nA_hat = \n", A_hat, type(A_hat))
	L_A_loss = torch.abs(A_hat - torch.eye(A_hat.shape[0])).sum().cuda() \
		if torch.cuda.is_available() else torch.abs(A_hat - torch.eye(A_hat.shape[0])).sum().cpu()
	print("L_A_loss = ", L_A_loss)
	return A_hat


####!!!! DEPRECATED
def gen_correlation_np(A):
	A = A.detach().numpy()
	I_c = np.eye(A.shape[0])
	# print(I_c)
	Awave = A + I_c
	tmp = Awave.sum(1)
	# print(tmp)
	D_wave_negative_power = np.diag(tmp ** (-0.5))
	# print(D_wave_negative_power)
	Ahat = np.matmul(np.matmul(D_wave_negative_power, Awave), D_wave_negative_power)
	# print(Ahat)
	Ahat = torch.from_numpy(Ahat)
	Ahat.clone().detach().requires_grad_(True)
	Ahat = Ahat.to(torch.float32)
	# print("Ahat = ", Ahat, Ahat.dtype)
	L_A_loss = torch.abs(Ahat - torch.eye(Ahat.shape[0])).sum().cuda() \
		if torch.cuda.is_available() else torch.abs(Ahat - torch.eye(Ahat.shape[0])).sum().cpu()
	print("L_A_loss=", L_A_loss)
	return Ahat


#
if __name__ == "__main__":
	# voc_adj.pkl path
	dir_voc_adj = "./data/voc/voc_adj.pkl"
	import pickle
	
	result = pickle.load(open(dir_voc_adj, 'rb'))
	_adj = result['adj']
	A_x = torch.tensor([[13.8252, 12.9317, 12.7645, 12.7931, 12.6785, 12.9123, 12.8959, 12.7716,
						 12.7385, 12.8504, 12.7110, 12.8148, 12.8755, 12.9428, 12.7443, 12.8226,
						 12.8516, 12.8119, 12.9014, 12.7620],
						[12.8325, 13.9444, 12.7748, 12.7979, 12.6815, 12.9140, 12.9071, 12.7734,
						 12.7483, 12.8613, 12.7219, 12.8271, 12.8834, 12.9483, 12.7601, 12.8386,
						 12.8545, 12.8158, 12.9051, 12.7658],
						[12.8486, 12.9755, 13.7952, 12.8242, 12.7129, 12.9410, 12.9350, 12.7979,
						 12.7741, 12.8849, 12.7434, 12.8470, 12.9028, 12.9770, 12.7729, 12.8625,
						 12.8751, 12.8491, 12.9276, 12.7882],
						[12.8644, 12.9702, 12.7975, 13.8290, 12.6993, 12.9381, 12.9363, 12.7968,
						 12.7696, 12.8920, 12.7383, 12.8494, 12.9134, 12.9775, 12.7714, 12.8601,
						 12.8822, 12.8371, 12.9304, 12.7839],
						[12.8148, 12.9171, 12.7503, 12.7728, 13.6631, 12.8914, 12.8854, 12.7489,
						 12.7242, 12.8332, 12.6933, 12.8022, 12.8607, 12.9242, 12.7344, 12.8083,
						 12.8297, 12.7922, 12.8864, 12.7385],
						[12.8135, 12.9273, 12.7459, 12.7776, 12.6625, 13.8971, 12.8882, 12.7575,
						 12.7315, 12.8440, 12.6938, 12.8103, 12.8680, 12.9298, 12.7335, 12.8176,
						 12.8364, 12.7909, 12.8864, 12.7428],
						[12.8045, 12.9096, 12.7352, 12.7590, 12.6535, 12.8824, 13.8730, 12.7441,
						 12.7165, 12.8310, 12.6800, 12.7934, 12.8536, 12.9166, 12.7248, 12.8025,
						 12.8277, 12.7786, 12.8733, 12.7248],
						[12.7783, 12.8909, 12.7136, 12.7441, 12.6323, 12.8569, 12.8529, 13.7107,
						 12.6862, 12.8022, 12.6613, 12.7627, 12.8222, 12.8936, 12.6983, 12.7727,
						 12.7890, 12.7629, 12.8489, 12.7020],
						[12.8183, 12.9338, 12.7509, 12.7891, 12.6762, 12.9063, 12.9024, 12.7601,
						 13.7370, 12.8495, 12.7051, 12.8091, 12.8677, 12.9412, 12.7392, 12.8176,
						 12.8450, 12.8020, 12.8968, 12.7615],
						[12.8457, 12.9775, 12.7922, 12.8269, 12.7147, 12.9370, 12.9372, 12.7991,
						 12.7668, 13.8818, 12.7377, 12.8519, 12.9043, 12.9799, 12.7820, 12.8541,
						 12.8726, 12.8382, 12.9289, 12.7924],
						[12.8399, 12.9574, 12.7823, 12.8111, 12.6994, 12.9202, 12.9158, 12.7867,
						 12.7564, 12.8814, 13.7273, 12.8368, 12.8960, 12.9606, 12.7689, 12.8419,
						 12.8733, 12.8293, 12.9142, 12.7698],
						[12.7853, 12.9000, 12.7217, 12.7518, 12.6333, 12.8672, 12.8636, 12.7210,
						 12.6969, 12.8170, 12.6664, 13.7708, 12.8333, 12.9064, 12.7107, 12.7838,
						 12.8017, 12.7723, 12.8631, 12.7091],
						[12.8531, 12.9643, 12.7804, 12.8136, 12.6999, 12.9245, 12.9223, 12.7869,
						 12.7599, 12.8714, 12.7316, 12.8350, 13.8907, 12.9699, 12.7662, 12.8394,
						 12.8637, 12.8321, 12.9176, 12.7665],
						[12.7848, 12.8963, 12.7180, 12.7444, 12.6278, 12.8646, 12.8584, 12.7201,
						 12.6955, 12.8068, 12.6672, 12.7731, 12.8385, 13.8990, 12.7087, 12.7791,
						 12.8032, 12.7642, 12.8585, 12.7089],
						[12.7788, 12.8903, 12.7188, 12.7486, 12.6269, 12.8592, 12.8529, 12.7180,
						 12.6906, 12.8106, 12.6610, 12.7680, 12.8339, 12.9013, 13.7021, 12.7829,
						 12.8054, 12.7587, 12.8531, 12.7015],
						[12.8617, 12.9769, 12.8054, 12.8350, 12.7151, 12.9477, 12.9362, 12.8226,
						 12.7756, 12.9043, 12.7429, 12.8659, 12.9220, 12.9777, 12.7843, 13.8657,
						 12.8971, 12.8370, 12.9357, 12.7964],
						[12.8138, 12.9275, 12.7522, 12.7885, 12.6700, 12.8951, 12.8878, 12.7527,
						 12.7246, 12.8376, 12.6972, 12.8038, 12.8587, 12.9366, 12.7365, 12.8097,
						 13.8363, 12.8006, 12.8863, 12.7428],
						[12.8231, 12.9452, 12.7619, 12.7984, 12.6938, 12.9138, 12.9050, 12.7736,
						 12.7495, 12.8641, 12.7187, 12.8200, 12.8793, 12.9521, 12.7550, 12.8304,
						 12.8558, 13.8184, 12.9013, 12.7680],
						[12.8960, 13.0074, 12.8287, 12.8623, 12.7469, 12.9776, 12.9688, 12.8448,
						 12.8059, 12.9245, 12.7737, 12.8919, 12.9477, 13.0117, 12.8131, 12.8970,
						 12.9174, 12.8722, 13.9667, 12.8205],
						[12.8363, 12.9511, 12.7737, 12.8063, 12.6870, 12.9232, 12.9250, 12.7812,
						 12.7496, 12.8744, 12.7223, 12.8326, 12.8911, 12.9616, 12.7599, 12.8467,
						 12.8641, 12.8192, 12.9108, 13.7706]], )
	# y = gen_A(0.15, 20, 0.4, str(dir_voc_adj))
	# # print(y)
	# z = gen_adj(torch.from_numpy(y).float())
	# gen_correlation(torch.from_numpy(_adj))
	# gen_correlation(A_x)
	x = torch.tensor([[2.5980, 1.5703, 1.5802, 1.5755, 1.5878, 1.5680, 1.5527, 1.5905, 1.5420,
					   1.5804, 1.5751, 1.5637, 1.5732, 1.5566, 1.5909, 1.5648, 1.5821, 1.5907,
					   1.5892, 1.5774],
					  [1.6007, 2.5733, 1.5807, 1.5772, 1.5903, 1.5732, 1.5558, 1.5920, 1.5432,
					   1.5846, 1.5755, 1.5663, 1.5773, 1.5572, 1.5925, 1.5687, 1.5848, 1.5880,
					   1.5915, 1.5780],
					  [1.5873, 1.5632, 2.5734, 1.5661, 1.5787, 1.5602, 1.5443, 1.5814, 1.5305,
					   1.5730, 1.5664, 1.5536, 1.5657, 1.5491, 1.5794, 1.5571, 1.5718, 1.5801,
					   1.5813, 1.5675],
					  [1.6297, 1.6060, 1.6140, 2.6053, 1.6217, 1.5998, 1.5824, 1.6234, 1.5739,
					   1.6159, 1.6047, 1.5958, 1.6089, 1.5883, 1.6197, 1.5976, 1.6158, 1.6215,
					   1.6218, 1.6060],
					  [1.6106, 1.5845, 1.5949, 1.5892, 2.6019, 1.5882, 1.5695, 1.6092, 1.5571,
					   1.5984, 1.5869, 1.5813, 1.5900, 1.5722, 1.6027, 1.5797, 1.5999, 1.6032,
					   1.6060, 1.5948],
					  [1.6111, 1.5900, 1.6013, 1.5912, 1.6078, 2.5872, 1.5709, 1.6102, 1.5607,
					   1.6026, 1.5910, 1.5853, 1.5962, 1.5731, 1.6098, 1.5837, 1.6020, 1.6040,
					   1.6073, 1.5948],
					  [1.6029, 1.5774, 1.5888, 1.5806, 1.5968, 1.5766, 2.5597, 1.5963, 1.5470,
					   1.5909, 1.5819, 1.5716, 1.5832, 1.5618, 1.5980, 1.5731, 1.5900, 1.5942,
					   1.5969, 1.5831],
					  [1.6043, 1.5799, 1.5899, 1.5827, 1.5973, 1.5808, 1.5611, 2.5984, 1.5509,
					   1.5916, 1.5804, 1.5716, 1.5826, 1.5644, 1.5970, 1.5726, 1.5911, 1.5966,
					   1.6010, 1.5840],
					  [1.5989, 1.5737, 1.5809, 1.5770, 1.5910, 1.5734, 1.5568, 1.5936, 2.5433,
					   1.5843, 1.5741, 1.5661, 1.5788, 1.5597, 1.5913, 1.5697, 1.5890, 1.5890,
					   1.5933, 1.5800],
					  [1.5909, 1.5668, 1.5799, 1.5692, 1.5860, 1.5675, 1.5511, 1.5883, 1.5397,
					   2.5798, 1.5730, 1.5604, 1.5718, 1.5513, 1.5878, 1.5623, 1.5790, 1.5845,
					   1.5866, 1.5746],
					  [1.5893, 1.5608, 1.5710, 1.5659, 1.5788, 1.5613, 1.5444, 1.5834, 1.5329,
					   1.5735, 2.5641, 1.5541, 1.5665, 1.5470, 1.5779, 1.5571, 1.5742, 1.5793,
					   1.5798, 1.5688],
					  [1.5810, 1.5581, 1.5673, 1.5578, 1.5733, 1.5564, 1.5392, 1.5736, 1.5264,
					   1.5674, 1.5583, 2.5477, 1.5598, 1.5420, 1.5731, 1.5509, 1.5667, 1.5732,
					   1.5761, 1.5579],
					  [1.5653, 1.5409, 1.5519, 1.5424, 1.5568, 1.5383, 1.5242, 1.5591, 1.5104,
					   1.5526, 1.5425, 1.5316, 2.5433, 1.5248, 1.5590, 1.5366, 1.5511, 1.5573,
					   1.5598, 1.5433],
					  [1.6107, 1.5849, 1.5945, 1.5867, 1.6018, 1.5834, 1.5670, 1.6022, 1.5544,
					   1.5971, 1.5881, 1.5769, 1.5880, 2.5694, 1.6030, 1.5802, 1.5955, 1.6028,
					   1.6030, 1.5909],
					  [1.5895, 1.5679, 1.5794, 1.5709, 1.5839, 1.5688, 1.5514, 1.5883, 1.5386,
					   1.5796, 1.5695, 1.5622, 1.5712, 1.5530, 2.5877, 1.5619, 1.5804, 1.5839,
					   1.5870, 1.5729],
					  [1.6108, 1.5876, 1.5994, 1.5895, 1.6023, 1.5885, 1.5706, 1.6057, 1.5572,
					   1.5969, 1.5908, 1.5783, 1.5878, 1.5720, 1.6060, 2.5826, 1.5986, 1.6035,
					   1.6078, 1.5940],
					  [1.6013, 1.5781, 1.5898, 1.5789, 1.5953, 1.5762, 1.5599, 1.5978, 1.5490,
					   1.5921, 1.5812, 1.5694, 1.5822, 1.5606, 1.5965, 1.5747, 2.5914, 1.5937,
					   1.5969, 1.5837],
					  [1.5731, 1.5479, 1.5554, 1.5510, 1.5638, 1.5456, 1.5316, 1.5674, 1.5174,
					   1.5588, 1.5503, 1.5395, 1.5520, 1.5355, 1.5643, 1.5425, 1.5586, 2.5629,
					   1.5667, 1.5533],
					  [1.5996, 1.5778, 1.5882, 1.5787, 1.5950, 1.5744, 1.5596, 1.5954, 1.5497,
					   1.5920, 1.5814, 1.5718, 1.5828, 1.5618, 1.5979, 1.5725, 1.5895, 1.5951,
					   2.5945, 1.5838],
					  [1.6009, 1.5795, 1.5922, 1.5801, 1.5976, 1.5772, 1.5600, 1.6006, 1.5516,
					   1.5937, 1.5824, 1.5748, 1.5854, 1.5637, 1.5982, 1.5731, 1.5915, 1.5956,
					   1.5967, 2.5842]])
	
	y = torch.tensor([[0.1754, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.1753, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.1759, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.1737, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.1746, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1744, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1750, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1749, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1752,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.1755, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.1759, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.1762, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.1771, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.1747, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1756, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1745, 0.0000, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1750, 0.0000,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1767,
					   0.0000, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.1750, 0.0000],
					  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
					   0.0000, 0.1749]])
	
	# print(torch.matmul(x,y))
	
	gen_correlation(x)
