import torch as th
import numpy as np
import sys, os

class CauchyLoss(th.nn.Module):
	def __init__(self, gamma=1, q_lambda=0.55, sij_type="IOU", normed=True):
		super(CauchyLoss, self).__init__()
		self.q_loss_img, self.cos_loss, self.loss = 0, 0, 0
		self.gamma = gamma
		self.q_lambda = q_lambda
		self.sij_type = sij_type
		self.normed = normed
		self.gpu_state = th.cuda.is_available()
		
	def forward(self, batch_label,  layer_out, LOCAL_TRAINING_FLAG=True):
		'''
		calculate the loss value
		:param self:
		:return:
		'''
		self.b_label = batch_label
		self.output_dim = layer_out.shape[1]       # hash bit value  2,4,8,16,32,64,128 ...
		self.u = layer_out.float()                  # u is the last layer output , they are float value
		self.label_u = batch_label.float()          # this is ground-truth label of all imgs in a batch
		# if self.training:
			# print("CauchyLoss on_forward :Trap into the train criterion")
		return self.apply_loss_function()
	 
	def cauchy_cross_entropy(self, v=None, label_v=None):
		label_u = self._gpu_cpu(self.label_u)
		u = self._gpu_cpu(self.u)
		normed = self.normed
		
		if v is None:
			v = self._gpu_cpu(u)
			label_v = self._gpu_cpu(label_u)
		
		# clip every element range from min_value to max_value, norm every element into range from [0.0, 1.0]
		# doc link-https://www.tensorflow.org/api_docs/python/tf/clip_by_value
		
		s = self._gpu_cpu(self.SIJ_RES(label_u, res_type=self.sij_type))
		# print("cauchy_cross_entropy s=\n{0}\n,s.shape=\n{1}\n" \
		#       .format(s, s.shape))
		# print()
		
		###### process output vector by fc8 layer ##################
		if normed:
			ip_1 = self._gpu_cpu(th.matmul(u, th.transpose(v, 0, 1)).float())  # u * v_T)
			# print("cauchy_cross_entropy mormed=true ip_1=\n",ip_1)
			# print()
			
			def reduce_shaper(t):
				return th.reshape(th.sum(t, 1), [t.shape[0], 1])
			
			# cosine value equals to u*v / sqrt(|u||u||v||v|)
			mod_1 = self._gpu_cpu(th.sqrt(th.matmul(reduce_shaper(u.pow(2)), th.transpose(reduce_shaper(v.pow(2)) + 0.000001,0,1))).float())
			# print("cauchy_cross_entropy mormed=true mod_1=\n{0}\n,mod_1.shape=\n{1}\n" \
			#       .format(mod_1, mod_1.shape))
			# print()
			
			# according to the formula(6) of the DCH paper, cosine value equals to u*v / sqrt(|u||u||v||v|)
			# print("cauchy_cross_entropy mormed=true th.div(ip_1, mod_1)=\n",th.div(ip_1, mod_1))
			# print()
			dist = self._gpu_cpu(float(self.output_dim) / 2.0 * (1.0 - th.div(ip_1, mod_1) + 0.000001))
			# dist = 64.0 / 2.0 * (1.0 - th.div(ip_1, mod_1) + 0.000001)            # test use this
			# print("cauchy_cross_entropy mormed=true dist=\n{0}\n,dist.shape=\n{1}\n" \
			#       .format(dist, dist.shape))
			# print()
		else:
			# if not normed ,then use the formula (6) to calculate
			# reduce_sum doc link-https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum
			r_u = self._gpu_cpu(th.reshape(th.sum(u * u, 1), [-1, 1]))  # u*u equal the tf.square(u))
			# print("cauchy_cross_entropy mormed=false r_u=\n{0}\n,r_u.shape=\n{1}\n" \
			#       .format(r_u, r_u.shape))
			# print()
			r_v = self._gpu_cpu(th.reshape(th.sum(v * v, 1), [-1, 1]))
			# print("cauchy_cross_entropy mormed=false r_v=\n{0}\n,r_v.shape=\n{1}\n" \
			#       .format(r_v, r_v.shape))
			# print()
			
			# u*u - 2*u*v + v*v + 0.001,e.g. (vector_u - vector_v)**2
			dist = self._gpu_cpu(r_u - 2 * th.matmul(u, th.transpose(v, 0, 1)) + th.transpose(r_v, 0, 1) + 0.001)
			# print("cauchy_cross_entropy mormed=false dist=\n{0}\n,dist.shape=\n{1}\n" \
			#       .format(dist, dist.shape))
			# print()
		
		# according to the formula(4) of the DCH paper
		cauchy = self._gpu_cpu(self.gamma / (dist + self.gamma))
		# print("cauchy_cross_entropy mormed=false cauchy=\n{0}\n,cauchy.shape=\n{1}\n" \
		#       .format(cauchy, cauchy.shape))
		# print()
		
		##### process label ##########
		# (s-0.5) * 2
		s_t = self._gpu_cpu(2.0 * th.add(s, -0.5)) # 2*s - 1      s_t shape [b_s,b_s]
		# print("cauchy_cross_entropy s_t=\n{0}\n" \
		#       .format(s_t))
		sum_1 = float(th.sum(s))
		# print("cauchy_cross_entropy sum_1=\n{0}\n" \
		#       .format(sum_1))
		sum_all = float(th.sum(th.abs(s_t)))
		# print("cauchy_cross_entropy sum_all=\n{0}\n" \
		#       .format(sum_all))
		# |s-1.0| + s * (sum_all/sum_1), this is the "weight" in formula (2)
		assert sum_1!=0, 'Maybe the batch size too small, so the label vectors have no interact set'
		balance_param = self._gpu_cpu(th.add(th.abs(th.add(s, -1.0)), float(sum_all / sum_1) * s))     # shape [b_s, b_s]
		# print("cauchy_cross_entropy balance_param=\n{0}\n,balance_param,shape=\n{1}\n" \
		#       .format(balance_param, balance_param.shape))
		
		# equal function: Returns the truth value of (x == y) element-wise.
		# get a matrix with all the Non-main diagonal elements are all "True"
		# main diagonal elements are all "False",
		mask = self._gpu_cpu(self.create_mask(s.shape[0])).long()
		# print("mask=\n{0}\n".format(mask))
		# boolean_mask function : Apply boolean mask to tensor
		cauchy_mask = self._gpu_cpu(th.gather(cauchy,1,mask).reshape(1,-1).squeeze())     # set diagonal elements as "False"
		# print("cauchy_mask=\n{0}\n,cauchy=\n{1}\n".format(cauchy_mask,cauchy))
		s_mask = self._gpu_cpu(th.gather(s,1,mask).reshape(1,-1).squeeze())
		# print("s_mask=\n{0}\n,s=\n{1}\n".format(s_mask, s))
		balance_p_mask = self._gpu_cpu(th.gather(balance_param,1,mask).reshape(1,-1).squeeze())     # shape -> [
		# print("balance_p_mask=\n{0}\n,balance_param=\n{1}\n".format(balance_p_mask, balance_param))
		
		# corresponding to the formula(8) in the DCH paper
		all_loss = self._gpu_cpu(- s_mask * th.log(cauchy_mask) - (1.0 - s_mask) * th.log(1.0 - cauchy_mask))
		# print("all_loss=\n{0}\n,all_loss.shape=\n{1}\n".format(all_loss, all_loss.shape))
		
		# print("th.matmul(all_loss, balance_p_mask=",th.mul(all_loss, balance_p_mask))
		# reduce_mean function :Computes the mean of elements across dimensions of a tensor.
		# print("th.mean(th.mul(all_loss, balance_p_mask))=",th.mean(th.mul(all_loss, balance_p_mask)))
		# print()
		return self._gpu_cpu(th.mean(th.mul(all_loss, balance_p_mask)))
	 
	def apply_loss_function(self):
		### loss function
		self.cos_loss = self.cauchy_cross_entropy()
		# print("torch edition apply_loss_function:self.cos_loss=\n",self.cos_loss)
		# print()
		
		self.q_loss_img = th.mean(th.pow(th.abs(self.u)-1.0,2))
		# print("torch edition apply_loss_function:self.q_loss_img=\n",self.q_loss_img)
		# print()
		# lambda * q_loss_img
		self.q_loss = self.q_lambda * self.q_loss_img
		# print("torch edition apply_loss_function:self.q_loss=\n",self.q_loss)
		# print()

		# L + lambda * Q
		self.loss = self.cos_loss + self.q_loss
		# print("torch edition apply_loss_function:self.loss=\n",self.loss)
		# print()
		
		return self.loss
	
	def test_criterion(self):
		'''
		when test ,use this function
		:return:
		'''
		pass
	 
	def clip_by_tensor(self, t, t_min, t_max):
		"""
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
		t = t.float()
		# result = (t >= t_min).float() * t + (t < t_min).float() * t_min
		# result = (result <= t_max).float() * result + (result > t_max).float() * t_max
		result = th.clamp(t, min=0.0, max=1.0)
		return result
		
	def SIJ_RES(self, lm_1, lm_2=None, res_type="IOU"):
		"""
        :param lm: 标签形成的one-hot
        :param res_type: IOU表示交集除以并集 其他情况为有交集为1，无交集为0
        :return: sij，自己与自己为1
        """
		if lm_2==None:
			lm_2 = lm_1
		
		dim = lm_1.shape[0]
		silm = self._gpu_cpu(th.matmul(lm_1, th.transpose(lm_2,0,1)).float())
		# print(silm)
		em = self._gpu_cpu(th.eye(dim).float())
		dig_silm = self._gpu_cpu(th.mul(silm, em).float())  # main diagonal element denotes how many labels the i-th img includes)
		cdig_silm = self._gpu_cpu(silm - dig_silm)  # set the main diagonal elements as 0
		
		if res_type == "IOU":
			for i in range(dim):
				for j in range(i + 1, dim):
					cdig_silm[i][j] = cdig_silm[i][j] / (dig_silm[i][i] + dig_silm[j][j] - cdig_silm[i][j])
					cdig_silm[j][i] = cdig_silm[i][j]
			return self._gpu_cpu(cdig_silm + em)
		elif res_type =="original":
			label_ip = th.matmul(lm_1, th.transpose(lm_2, 0, 1)).float()  # label_u * label_v_T
			s = th.clamp(label_ip, min=0.0, max=1.0)
			# s[s <= 0] = 0.0
			# s[s > 0] = 1.0
			return self._gpu_cpu(s)
		else:
			assert (res_type=="IOU" or res_type=="original"),\
				"No such initialize s_ij method... Process Abort!!!"
			# return self.clip_by_tensor(cdig_silm, 0.0, 1.0) + em  # 自己和自己也视为相似 其余有标签交集就代表相似
			# transform every elements of a variable into another type
	 
	def create_mask(self,mat_dim):
		'''
		create a mask matrix for torch.gather() operation
		:param mat_dim:
		:return:
		'''
		out = []
		out = th.zeros([mat_dim,mat_dim - 1]).long()
		pool = [x for x in range(mat_dim)]
		tmp_pool = []
		for i in range(mat_dim):
			tmp_pool = pool.copy()
			tmp_pool.remove(int(i))
			for j in range(mat_dim - 1):
				out[i][j] = tmp_pool[j]
				
		return self._gpu_cpu(out)
	
	def _gpu_cpu(self, input):
		'''
		adapt gpu or cpu environment
		:param input:
		:return:
		'''
		if th.is_tensor(input):
			if self.gpu_state:
				input = input.float()
				return input.cuda()
		return input.float()

if __name__ == "__main__":
	label_u = th.tensor([[1, 0, 0, 0, 0, 1], [1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0]]).float()
	
	u = th.tensor([[-1.8097, -1.8097, -1.8097,  1.4954,  1.5297, -1.9307],
	              [-1.8097, -1.8097, -1.8097,  1.4269,  1.4783, -1.4230],
	              [-1.9307, -1.9307, -1.9482,  -1.4405, -1.4580, 1.8034],
	              [-1.7347, -1.7347, -1.7522,  -1.4210, -1.4384, -1.7347]]).float()
	calcloss = CauchyLoss( sij_type='original',normed=False)
	# calcloss.cauchy_cross_entropy(normed=True,res_type="original")
	calcloss(label_u,u)

