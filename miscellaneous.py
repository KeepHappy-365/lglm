# -*- coding=utf-8 -*-
###############################################################################
# process many log files
# for example:
#   painting the loss-curve, mAP-curve
###############################################################################

import os, sys, matplotlib
import matplotlib.pyplot as plt

class processTXT:

	def __init__(self, txtFilePath, state={}):
		self.state = state
		self.txtFile = txtFilePath

		if self._state('chunkSize') is None:
			self.state['chunkSize'] = 4096*1024

	def _state(self, name):
		if str(name) in self.state:
			return self.state[str(name)]

	def read_by_chunk(self,):
		filepath = self.txtFile
		with open(filepath, 'r') as f:
			while True:
				chunk_data = f.read(4096*1024)  # every chunk = 4MB
				if not chunk_data:
					break
				yield  chunk_data

	def read_whole_display(self,):
		x = self.read_by_chunk()
		count = 0
		for chunk in x:
			count += 1
			y = chunk.split(' ')
			print( '\n', type(chunk), '\n', len(chunk))
			print( y, '\n', type(y), '\n', len(y))
			# for i in range(len(y)):
			# 	print('y[{0}] = {1}'.format(i, y[i]))
			break

		print("count = {count}\n".format(count=count))


	def extract_mllog(self, row_patter_str, patter_str, split_str="\n"):
		'''
		only used for extracting the ML task LOG files
		:param row_patter_str:  remark the interesting line
		:param patter_str:      remark the interesting mark
		:param split_str:       use which str for split the original STRING
		:return:
		'''
		result = []
		SPLIT_POOL = ['\t', ' ', '\n', ]
		x = self.read_by_chunk()
		count = 0
		for chunk in x:
			count += 1
			y = chunk.split(str(split_str))
			# print('\n', type(chunk), '\n', len(chunk))
			# print(y, '\n', type(y), '\n', len(y))
			for i in range(len(y)):
				# print('y[{0}] = {1}, type(y[])={2}'.format(i, y[i], type(y[i])))
				if row_patter_str in y[i]:
					if type(y[i])==type(' '):
						patter_str_index = y[i].find(str(patter_str), 0, len(y[i]))
						if patter_str_index != -1:
							item_split_str = y[i][patter_str_index+len(patter_str):patter_str_index+len(patter_str)+1]
							tmp = y[i].split(str(item_split_str))
							# print('tmp = ', tmp)
							for idx in range(len(tmp)):
								if patter_str in tmp[idx]:
									try:
										des = tmp[idx+1]
										des_tmp = ''
										if ' ' in des:
											des = des.strip(' ')
										if '\t' in des :
											des_list = des.split('\t')
											# print('des_list = ', des_list)
											des = des_list[0]
										result.append(float(des))
									except:
										print('The index out of list range...\n')

		# print(result)
		return result


class paintingFig:

	def __init__(self, flag='show', save_fig='./default'):
		self.flag = flag
		self.save_fig = save_fig

	def epoch_fig(self,epoch, **y_value):
		# print("len(y_value)=",len(y_value))
		color_pool = ['red', 'green', 'blue', 'black']
		save_fig = self.save_fig
		plt.figure(figsize=(15,10), dpi=80)
		if len(y_value) and len(epoch) :
			plt.title(str(save_fig))
			i = 0
			for k, v in y_value.items():
				plt.plot(epoch, v, marker="o", color=str(color_pool[i]), label=str(k))
				i += 1
			plt.grid(True)
			plt.legend(loc='lower right')

			if self.flag == 'show':
				plt.show()

			if self.flag == 'save':
				plt.savefig(save_fig + '.jpg')
				plt.clf()  # clear plt
		else:
			return

if __name__ == "__main__":
	filepath = '/home/xyz/Documents/HASH_CODE_important/ML_SGCN4TNNLS/data/nuswide/AllLabels81.txt'
	logpath = '/home/xyz/Documents/HASH_CODE_important/ML_GCN-Modified/voc_log_files/lah_voc_202006301145_closeMFB_closeMLfc'
	obj = processTXT(logpath)
	result1 = obj.extract_mllog("Test",'mAP')
	logpath = '/home/xyz/Documents/HASH_CODE_important/ML_GCN-Modified/voc_log_files/lah_voc_202006282310_closeMFB_openMLfc'
	obj = processTXT(logpath)
	result2 = obj.extract_mllog('Test', 'mAP')
	logpath = '/home/xyz/Documents/HASH_CODE_important/ML_GCN-Modified/voc_log_files/lah_202006302140_openMFB_openMLfc'
	obj = processTXT(logpath)
	result3 = obj.extract_mllog('Test', 'mAP')
	print('result1 = ', result1)
	print('result2 = ', result2)
	print('result3 = ', result3)
	p_obj = paintingFig(flag='save', save_fig='Test mAP')
	p_obj.epoch_fig([int(i) for i in range(1, len(result1)+1)],
	                closeMFB_closeMLfc=result1, openMFB_openMLfc=result2, openMFB_closeMLfc=result3)
