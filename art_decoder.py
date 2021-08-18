'''
Deep animation decoder.
'''
import os, sys
import numpy as np
from tqdm import tqdm

import torch
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float
from sync_batchnorm import DataParallelWithCallback
from scipy.spatial import ConvexHull 
# from animate import normalize_kp

from animate import get_animation_region_params
from common.metrics import *
from common.bpg import BPGDecoder
from modules import decompress
#from common.curves import *
from common.tools import *
from copy import deepcopy
import imageio
import json
import yaml

class ART_Decoder:
	def __init__(self,config, checkpoint,relative, adapt_scale,cpu,thresh=32,qp = 20):
		self.filename = ''
		self.bitstream_path = ''
		self.result_video_path = ''
		self.result_metrics_path = ''
		self.config_path = config
		self.checkpoint_path = checkpoint
		self.generator = {}
		self.kp_detector = {}
		self.moov = {}
		self.qp = qp
		self.thresh = thresh
		self.cpu = cpu
		self.relative = relative
		self.adapt_scale = adapt_scale
		self.decoded_sequence = []
		self.source_buffer = []
		self.kp_driving_initial = {}
		self.loaded = []
		self.src_list = []
			
	def _load_metadata(self):
		with open(self.bitstream_path + "mdat.json", "r") as moov:
			self.moov = deepcopy(json.load(moov))
			
	def _load_src_frames(self, frame, buffer_frame = None, border = False):
		sources = []
		if border:
			buff_frame = torch.tensor(img_as_float(buffer_frame)[np.newaxis].astype(np.float32)).permute(0,3,1,2)	
			if not self.cpu:
				buff_frame = buff_frame.cuda()
				sources.append(buff_frame)
		source_frame = torch.tensor(img_as_float(frame)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		if not self.cpu:
			src_frame_tensor = source_frame.cuda()
			sources.append(src_frame_tensor)
		return sources
		
	def _load_buffer(self,kp_driving):
		'''extracts a segments of the buffer info and uses it to encode the next frame'''
		sources = []
		kp_sources = []
		if len(self.source_buffer)<2:
			for idx in range(len(self.source_buffer)):
				sources.append(self.source_buffer[idx][0])
				kp_sources.append(self.source_buffer[idx][1])
			kp_norm_list = self._kp_norm_list(kp_driving, kp_sources)
		else:
			for idx in range(2):
				sources.append(self.source_buffer[-(2-idx)][0])
				kp_sources.append(self.source_buffer[-(2-idx)][1])
			kp_norm_list = self._kp_norm_list(kp_driving, kp_sources)
		return sources, kp_sources, kp_norm_list
		
	def _kp_norm_list(self, kp_driving, kp_sources):
		kp_norm_list = []
		for kp in kp_sources:
			kp_norm = normalize_kp(kp_source=kp, kp_driving=kp_driving,kp_driving_initial=self.kp_driving_initial, use_relative_movement=self.relative, use_relative_jacobian=self.relative, adapt_movement_scale=self.adapt_scale)
			kp_norm_list.append(kp_norm)
		return kp_norm_list
		
	def _smoothing_frame(self):
		'''Adds the latest decoded frame for smoothing to ensure spatial consistency in the decoded sequence'''
		src = torch.tensor(self.decoded_sequence[-1][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		if not self.cpu:
			src = src.cuda()
			kp_src= self.region_predictor(src)
		return ([src], [kp_src])
		
		
	def _load_assets(self, src_idx, generated=None):
		decoder  = BPGDecoder()
		if src_idx not in self.loaded:
			src_frame = decoder.decode(self.bitstream_path+str(src_idx)+'.bpg')
			if generated is not None:
				src_frame += generated
			src_frame = np.clip(src_frame,0,1)
			self.decoded_sequence.append(src_frame)
			source_frame = torch.tensor(src_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
			if not self.cpu:
				source_frame = source_frame.cuda()
				kp_source = self.region_predictor(source_frame)
				# print(kp_source)
			self.source_buffer.append((source_frame, kp_source, src_idx))
			self.loaded.append(src_idx)
			#animate src frame
			kp_driving = kp_source
			self.kp_driving_initial = kp_source
			# kp_norm_list = self._kp_norm_list(kp_driving, [kp_source])				
			# decoded = self.generator([source_frame],[kp_source], kp_norm_list, 0.3)
			# new_region_params = get_animation_region_params(kp_source, kp_driving,
			# 													self.kp_driving_initial, avd_network=self.avd_network,
			# 													mode='standard')
			# decoded = self.generator(source_frame,kp_source, new_region_params)
			# generated_frame = np.transpose(decoded['prediction'].data.cpu().numpy(),[0, 2, 3, 1])[0]
			# self.decoded_sequence.append(src_frame)
		else:
			target_idx= self.loaded.index(src_idx)
			source_frame = self.source_buffer[target_idx][0]
			kp_source = self.source_buffer[target_idx][1]
		return [source_frame], [kp_source]
		
	def decode(self,bitstream_path, filename):
		self.bitstream_path = bitstream_path + str(filename)+'/bitstream/'
		self.output_path = bitstream_path
		#load models
		# self.generator, self.region_predictor,self.avd_network, self.resEncoder, self.resDecoder,\
		# self.respriorEncoder, self.respriorDecoder, self.bitEstimator_z, self.config_options =\
		# load_checkpoints(config_path=self.config_path, checkpoint_path=self.checkpoint_path, cpu=self.cpu)
		self.generator, self.region_predictor,self.avd_network, self.config_options = load_checkpoints(config_path=self.config_path, checkpoint_path=self.checkpoint_path, cpu=self.cpu)
		#load moov_box
		self._load_metadata()
		#load key points
		kp_list = {'affine': [], 'covar': [], 'shift': []} 
		# with gzip.GzipFile(self.bitstream_path + "/0_kp.npy.gz", "rb") as keypoints:
		# 	kp_list['value'] = np.load(keypoints, allow_pickle = True)
		# with gzip.GzipFile(self.bitstream_path + "/0_jc.npy.gz", "rb") as jacobians:
		# 	kp_list['jacobian'] = np.load(jacobians, allow_pickle = True)
		decompress.decompress(self.bitstream_path + "/0_sh.cp", self.bitstream_path + "/0_sh.npy.gz")
		decompress.decompress(self.bitstream_path + "/0_af.cp", self.bitstream_path + "/0_af.npy.gz")
		with open(self.bitstream_path + "/0_sh.npy.gz", "rb") as keypoints:
			kp_list['shift'] = np.load(keypoints, allow_pickle = True)
		with open(self.bitstream_path + "/0_af.npy.gz", "rb") as jacobians:
			kp_list['affine'] = np.load(jacobians, allow_pickle = True)
		with open(self.bitstream_path + "/0_res.npy", "rb") as jacobians:
			kp_list['res'] = np.load(jacobians, allow_pickle = True)
		# with gzip.GzipFile(self.bitstream_path + "/0_qu.npy.gz", "rb") as quant:
		# 	kp_list['quant'] = np.load(quant, allow_pickle = True)
		print(len(kp_list['shift']))
		sources = []
		kp_sources = []
		src_seq_list = sorted(set(self.moov['src_seq']))
		for idx in tqdm(range(int(self.moov['nalus']))):
			if idx == 0:
				sources, kp_sources = self._load_assets(idx)
				self.kp_driving_initial = kp_sources[-1]
				continue
			if idx > 0 and idx < len(kp_list['shift']):
				kp_list['shift'][idx] += kp_list['shift'][idx-1]
				kp_list['affine'][idx] += kp_list['affine'][idx-1]
				if idx < len(kp_list['res']):
					kp_list['res'][idx] += kp_list['res'][idx-1]
			
			kp_driving = {}
			kp_first = {}
			kp_first['shift'] =torch.tensor(kp_list['shift'][0])
			kp_first['affine'] = torch.tensor(kp_list['affine'][0])
			kp_first['covar'] = \
				torch.matmul(kp_first['affine'], kp_first['affine'].permute(0, 1, 3, 2))
			kp_driving['shift'] =torch.tensor(kp_list['shift'][idx])
			kp_driving['affine'] = torch.tensor(kp_list['affine'][idx])
			kp_driving['covar'] = \
				torch.matmul(kp_driving['affine'], kp_driving['affine'].permute(0, 1, 3, 2))
			# kp_norm_list = self._kp_norm_list(kp_driving, kp_sources)	
			new_region_params = get_animation_region_params(self.kp_driving_initial, kp_driving,
															kp_first, avd_network=self.avd_network,
															mode='standard')
				
			decoded = self.generator(sources[-1], source_region_params=kp_sources[-1], driving_region_params=new_region_params)
			# add
			# if idx < len(kp_list['res']):
			# 	compressed_feature_renorm = torch.tensor(kp_list['res'][idx])

			# 	recon_res = self.resDecoder(compressed_feature_renorm)
			# 	recon_image = decoded['prediction'] + recon_res

			# 	clipped_recon_image = recon_image.clamp(0., 1.)
			# 	generated_frame = np.transpose(clipped_recon_image.data.cpu().numpy(),[0, 2, 3, 1])[0]
			generated_frame = np.transpose(decoded['prediction'].data.cpu().numpy(),[0, 2, 3, 1])[0]

			if idx in src_seq_list:
				sources,kp_sources = self._smoothing_frame()
				src,kp_src = self._load_assets(idx, generated=generated_frame)
				sources.append(src[-1])
				kp_sources.append(kp_src[-1])
				self.kp_driving_initial = kp_sources[-1]
			else:
				self.decoded_sequence.append(generated_frame)
		imageio.mimsave(self.output_path +str(filename)+ '/video/'+ 'decoded_'+str(self.qp)+'_'+str(self.thresh)+'.mp4', [img_as_ubyte(frame) for frame in self.decoded_sequence], fps = 30)
