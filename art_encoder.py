'''
Deep animation Encoder with multiple coding modes
0- single source frame encoding - lowest bitrate: param: source frame QP level
1 - Best performing source in terms of PSNR. Source buffer controlled by a euclidian distance in the keypoint space: params: euclidian threshold, min PSNR performance
2 - Best performing source in terms of Perceptual loss. Source buffer controlled by a euclidian distance in the keypoint space: params: euclidian threshold, max Perp. loss
'''
#Basic modules
import os, sys
import torch
import numpy as np

#FIle I/O
import random
import imageio
#data handling
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float
import torch.nn.functional as F
from logger import draw_colored_heatmap
#Plots and metrics
from common.metrics import *
#from common.curves import *
from common.tools import *
from common.bpg import BPGEncoder, BPGDecoder
from animate import get_animation_region_params
#MonkeyNET tools
# from animate import normalize_kp
from scipy.spatial import ConvexHull 
from copy import deepcopy
from collections import deque

class ART_Encoder:
	def __init__(self, config, checkpoint, qp, relative, adapt_scale, cpu,thresh):
		self.config_path = config
		self.checkpoint_path = checkpoint
		self.config_options = {}
		self.qp = qp
		self.relative = relative
		self.adapt_scale = adapt_scale
		self.cpu = cpu
		self.moov = {}
		self.generator = {} #pretrained generator model
		self.region_predictor = {} #pretrained region_predictor
		self.bg_predictor = {} #pretrained region_predictor
		self.avd_network = {} #pretrained avd_network
		self.driving_video_path = ''
		self.result_video_path = ''
		self.result_metrics_path = ''
		self.filename = ''
		self.threshold = thresh
		self.bitstream_path = ''
		self.predictions = []
		self.original_video = np.array([])
		self.predictions = []
		self.src_idx = [] #index of source frames in sequence
		self.source_buffer = deque([], maxlen=100)
		self.kp_driving_initial = {}
		self.sources = []
		self.kp_sources = []
		self.driving_sequence = []
		self.total_bytes = 0
		self.nalus = 0
		self.kp_list = {'affine': [], 'covar': [], 'shift': [], 'res': []} 
		self.res_list = []
		
			
	def _make_animation(self, video_sequence, cpu=False, animation_mode='standard'):
		with torch.no_grad():
			predictions = []
			frames = len(video_sequence)
			if frames > 120:
				self.nalus = 120
			else:
				self.nalus =  frames
			self.moov["nalus"] = self.nalus
			self.source, self.kp_source, bytes = self._add_source_frame(0)
			self.total_bytes += write_nalu(self.bitstream_path, source_img=self.original_video[0], qp=self.qp, nalu=0)
			print(self.total_bytes)
			if not cpu:
				self.source = self.source.cuda()
			self.driving_sequence = torch.tensor(np.array(video_sequence)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
			# source_region_params = self.region_predictor(self.source)
			driving_region_params_initial = self.region_predictor(self.driving_sequence[:, :, 0])

			driving_region_params_initial['d'] *= 1024
			driving_region_params_initial['d'] = driving_region_params_initial['d'].round()
			driving_region_params_initial['d'] /= 1024.
			driving_region_params_initial['u'] *= 1024
			driving_region_params_initial['u'] = driving_region_params_initial['u'].round()
			driving_region_params_initial['u'] /= 1024.
			# driving_region_params_initial['affine'] *= 1024
			# driving_region_params_initial['affine'] = driving_region_params_initial['affine'].round()
			# driving_region_params_initial['affine'] /= 1024.
			driving_region_params_initial['covar'] = \
			torch.matmul(driving_region_params_initial['affine'], driving_region_params_initial['affine'].permute(0, 1, 3, 2))
			driving_region_params_initial['shift'] *= 256
			driving_region_params_initial['shift'] = driving_region_params_initial['shift'].round()
			driving_region_params_initial['shift'] /= 256.
			print(driving_region_params_initial['heatmap'].shape)
			print(driving_region_params_initial['shift'].shape)
			print(driving_region_params_initial['affine'].shape)
			print(driving_region_params_initial['covar'].shape)
			print(driving_region_params_initial['u'].shape)
			self.kp_list['affine'].append(driving_region_params_initial['affine'].cpu().detach().numpy())
			self.kp_list['covar'].append(driving_region_params_initial['covar'].cpu().detach().numpy())
			self.kp_list['shift'].append(driving_region_params_initial['shift'].cpu().detach().numpy())
			# self.kp_list['u'].append(driving_region_params_initial['u'][:,:,:,0].cpu().detach().numpy())
			# self.kp_list['d'].append(driving_region_params_initial['d'][:,0,0].cpu().detach().numpy())
			np.set_printoptions(threshold=np.inf)
			# print(driving_region_params_initial['heatmap'][0][0].cpu().numpy())
			source_heatmap = F.interpolate(driving_region_params_initial['heatmap'], size=(256,256))
			source_heatmap = np.transpose(source_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
			source_heatmap = draw_colored_heatmap(source_heatmap, plt.get_cmap('gist_rainbow'), (0, 0, 0))
			print(self.driving_sequence[:, :, 0][0].shape)
			temp_map = np.transpose(self.driving_sequence[:, :, 0].data.cpu().numpy(), [0, 2, 3, 1])
			work_dir = os.getcwd()+'/temp'
			source_heatmap = np.clip(source_heatmap,0,1)
			imageio.imwrite(work_dir+'/temp_s.jpg', img_as_ubyte(source_heatmap[0]),quality=100)
			imageio.imwrite(work_dir+'/temp_t.jpg', img_as_ubyte(temp_map[0]),quality=100)
			# source_region_params = driving_region_params_initial

			self.source_buffer.append((self.source, self.kp_source,0))
			# print(self.kp_source)
			src_seq = [0]
			for frame_idx in tqdm(range(self.driving_sequence.shape[2])):
				if frame_idx >= 120:
					break
				if frame_idx == 0:
					continue
				(driving_frame, driving_region_params) = self._extract_driving_frame(frame_idx)
				if not cpu:
					driving_frame = driving_frame.cuda()
				# driving_region_params['u'] = 0
				# driving_region_params['d'] = 0
				# driving_region_params['heatmap'] = 0
				# driving_region_params['shift'] *= 0
				driving_region_params['d'] *= 1024
				driving_region_params['d'] = driving_region_params['d'].round()
				driving_region_params['d'] /= 1024.
				
				driving_region_params['u'] *= 1024
				driving_region_params['u'] = driving_region_params['u'].round()
				driving_region_params['u'] /= 1024.
					
				driving_region_params['affine'] *= 1024
				driving_region_params['affine'] = driving_region_params['affine'].round()
				driving_region_params['affine'] /= 1024.
				driving_region_params['covar'] = \
				torch.matmul(driving_region_params['affine'], driving_region_params['affine'].permute(0, 1, 3, 2))
					
				driving_region_params['shift'] *= 256
				driving_region_params['shift'] = driving_region_params['shift'].round()
				driving_region_params['shift'] /= 256.
				# draw
				if frame_idx == 100:
					source_heatmap = F.interpolate(driving_region_params['heatmap'], size=(256,256))
					source_heatmap = np.transpose(source_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
					source_heatmap = draw_colored_heatmap(source_heatmap, plt.get_cmap('gist_rainbow'), (0, 0, 0))
					work_dir = os.getcwd()+'/temp'
					temp_map = np.transpose(self.driving_sequence[:, :, frame_idx].data.cpu().numpy(), [0, 2, 3, 1])
					source_heatmap = np.clip(source_heatmap,0,1)
					imageio.imwrite(work_dir+'/temp_s2.jpg', img_as_ubyte(source_heatmap[0]),quality=100)
					imageio.imwrite(work_dir+'/temp_t2.jpg', img_as_ubyte(temp_map[0]),quality=100)

				generated, idx, encoded = self._find_best_source_psnr(driving_region_params, frame_idx)
				if encoded:
					self.predictions.append(generated)
				
				self.kp_list['affine'].append(driving_region_params['affine'].cpu().detach().numpy())
				self.kp_list['covar'].append(driving_region_params['covar'].cpu().detach().numpy())
				self.kp_list['shift'].append(driving_region_params['shift'].cpu().detach().numpy())
				
				# self.kp_list['u'].append(driving_region_params['u'][:,:,:,0].cpu().detach().numpy())
				# self.kp_list['d'].append(driving_region_params['d'][:,0,0].cpu().detach().numpy())

				# new_region_params = get_animation_region_params(source_region_params, driving_region_params,
				# 												driving_region_params_initial, avd_network=self.avd_network,
				# 												mode=animation_mode)
				# out = self.generator(ref, source_region_params=source_region_params, driving_region_params=new_region_params)
				# predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
				if not encoded:
					print(frame_idx)
					# res_frame = self.original_video[frame_idx] - np.array(generated)
					# res_frame *= 256
					# for i in range(res_frame.shape[0]):
					# 	for j in range(res_frame.shape[1]):
					# 		for k in range(res_frame.shape[2]):
					# 			if res_frame[i,j,k] < 10:
					# 				res_frame[i,j,k] = 0
					# res_frame = res_frame.round()
					# res_frame /= 256.
					# # res_frame, _ = self._decode_frame(res_frame, frame_idx)
					# self.total_bytes += write_nalu(self.bitstream_path,res=res_frame,nalu=frame_idx)
					# self.res_list.append(res_frame)
				
				src_seq.append(idx)
			bytes = 458
			print(len(self.kp_list["affine"]))
			for i in range(len(self.kp_list["affine"])-2,-1,-1):
				self.kp_list["affine"][i+1] -= self.kp_list["affine"][i]
			# for i in range(len(kp_list["covar"])-2,-1,-1):
			#     kp_list["covar"][i+1] -= kp_list["covar"][i]
			for i in range(len(self.kp_list["shift"])-2,-1,-1):
				self.kp_list["shift"][i+1] -= self.kp_list["shift"][i]
			for i in range(len(self.kp_list["res"])-2,-1,-1):
				self.kp_list["res"][i+1] -= self.kp_list["res"][i]
			# for i in range(len(self.kp_list["u"])-2,-1,-1):
			# 	self.kp_list["u"][i+1] -= self.kp_list["u"][i]
			# for i in range(len(self.kp_list["d"])-2,-1,-1):
			# 	self.kp_list["d"][i+1] -= self.kp_list["d"][i]
			
			self.moov['src_seq'] = src_seq
			self.total_bytes += write_nalu(self.bitstream_path, kp= self.kp_list)
			self.total_bytes += write_nalu(self.bitstream_path, moov_box= self.moov)
		print(self.total_bytes)
		imageio.mimsave('result.mp4', [img_as_ubyte(frame) for frame in self.predictions], fps=30)
		return predictions, bytes
	
	def _extract_driving_frame(self, idx):
		'''Uses the keypoint detector from the F-O-M to get driving frame kps'''
		driving_frame = self.driving_sequence[:, :, idx]
		if not self.cpu:
			driving_frame = driving_frame.cuda()
			kp_driving = self.region_predictor(driving_frame)
		return (driving_frame, kp_driving)	
		
	def _add_kp_list(self,kp_driving):
		#accumulates the driving kps for buffer-wise entropy coding
		self.kp_list['value'].append(kp_driving['value'])
		self.kp_list['jacobian'].append(kp_driving['jacobian'])
		# self.kp_list['quant'].append(kp_driving['quant'])

	def _kp_norm_list(self, kp_driving, kp_sources):
		kp_norm_list = []
		for kp in kp_sources:
			kp_norm = normalize_kp(kp_source=kp, kp_driving=kp_driving,kp_driving_initial=self.kp_driving_initial, use_relative_movement=self.relative, use_relative_jacobian=self.relative, adapt_movement_scale=self.adapt_scale)
			kp_norm_list.append(kp_norm)
		return kp_norm_list

	def _normalize_driving_kp(self, kp_driving, kp_source):
		kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,kp_driving_initial=self.kp_driving_initial, use_relative_movement=self.relative, use_relative_jacobian=self.relative, adapt_movement_scale=self.adapt_scale)
		return kp_norm
		
	def _animate(self, sources, kp_sources, kp_driving, frame_idx=0):
		'''Calls the Motion transfer network from the F-O-M to synthesize an animated frame given a source(list of sources) and keypoints from the driving frame'''
		# out = self.generator(sources,kp_sources, kp_norm_list,0.3)
		# generated = np.transpose(out['prediction'].data.cpu().numpy(),[0, 2, 3, 1])[0]
		new_region_params = get_animation_region_params(self.kp_driving_initial, kp_driving,
																self.kp_driving_initial, avd_network=self.avd_network,
																mode='standard')
		# print(kp_sources)			
		
		if not self.cpu:
			sources = sources.cuda()
			driving = self.driving_sequence[:, :, frame_idx].cuda()
		# bg_params = self.bg_predictor(sources, driving)	
		out = self.generator(sources, source_region_params=kp_sources, driving_region_params=new_region_params)
		# add
		# input_residual = driving - out['prediction']
		# # print(input_residual)

		# feature = self.resEncoder(input_residual)
		# batch_size = feature.size()[0]

		# z = self.respriorEncoder(feature)
		# # quant_noise_feature, quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(feature), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(z), -0.5, 0.5)

		# compressed_z = torch.round(z)

		# recon_sigma = self.respriorDecoder(compressed_z)

		# feature_renorm = feature
		# compressed_feature_renorm = torch.round(feature_renorm)
		
		# self.kp_list['res'].append(compressed_feature_renorm.cpu().detach().numpy())

		# recon_res = self.resDecoder(compressed_feature_renorm)
		# recon_image = out['prediction'] + recon_res

		# clipped_recon_image = recon_image.clamp(0., 1.)

		generated = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
		# generated = np.transpose(clipped_recon_image.data.cpu().numpy(), [0, 2, 3, 1])[0]
		return generated, out['occlusion_map']
		
	def _expand_source_buffer(self, idx, generated=None, o_map=None):
		'''selects a new source frame, adds it to buffer and encodes it directly into the bitstream'''
		src, kp, bytes = self._add_source_frame(idx, generated=generated, o_map=o_map)
		self.source_buffer.append((src, kp, idx))
		if len(self.source_buffer) > 3:
			self.source_buffer.popleft()
		#write source frame to bitstream
		# self.total_bytes += write_nalu(self.bitstream_path, source_img=self.original_video[idx]-generated, qp=self.qp, nalu=idx)
		
	def _add_source_frame(self, idx, source=True, generated=None, o_map=None):
		if generated is None:
			frame, bytes= self._decode_frame(self.original_video[idx], idx)
		else:
			res_frame = self.original_video[idx] - generated
			# res_frame *= 255
			# for i in range(res_frame.shape[0]):
			# 	for j in range(res_frame.shape[1]):
			# 		for k in range(res_frame.shape[2]):
			# 			if res_frame[i,j,k] < 4:
			# 				res_frame[i,j,k] = 0
			# res_frame = res_frame.round()
			# res_frame /= 255.
			# res_frame *= o_map[0]
			frame, bytes= self._decode_frame(res_frame, idx)
			self.total_bytes += write_nalu(self.bitstream_path,source_img=frame, qp=self.qp,nalu=idx)
			# frame, bytes= self._decode_frame(self.original_video[idx]-generated, idx)
			frame += generated
			frame = np.clip(frame,0,1)
		
		self.predictions.append(frame)
		src = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		# if source:
		# 	self.predictions.append(frame)
		if not self.cpu:
			src = src.cuda()
			kp_source = self.region_predictor(src)
		return src, kp_source,bytes

	def _decode_frame(self, target, idx):
		#synchronizes frame quality between encoder and decoder
		bpg_enc = BPGEncoder(qp=str(self.qp))
		work_dir = os.getcwd()+'/temp'
		try:	
			os.mkdir(work_dir)
		except:
			pass
		imageio.imwrite(work_dir+'/temp_'+str(idx)+'.jpg', img_as_ubyte(target),quality=100)
		frame, bytes = bpg_enc.encode(filename = work_dir+'/temp_'+str(idx)+'.jpg', output= work_dir+'/temp', out=True)
		# os.remove(work_dir+'/temp_'+str(idx)+'.jpg')
		return frame, bytes
		
	def _sync_as_source(self, idx):
		frame, bytes = self._decode_frame(self.original_video[idx], idx)
		source_frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
		if not self.cpu:
			source_frame = source_frame.cuda()
			kp_source = self.kp_detector(source_frame)
		#animate src frame
		kp_driving = kp_source
		self.kp_driving_initial = kp_source
		kp_norm_list = self._kp_norm_list(kp_driving, [kp_source])				
		decoded = self.generator([source_frame],[kp_source], kp_norm_list, 0.3)
		decoded_src = np.transpose(decoded['prediction'].data.cpu().numpy(),[0, 2, 3, 1])[0]
		psnr_val = psnr(img_as_ubyte(self.original_video[idx]), img_as_ubyte(decoded_src))
		return psnr_val, bytes	
	
	def _smoothing_frame(self):
		'''Adds the latest decoded frame for smoothing to ensure spatial consistency in the decoded sequence'''
		src = torch.tensor(self.predictions[-1][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
		if not self.cpu:
			src = src.cuda()
			kp_source = self.region_predictor(src)
		return (src,kp_source)		
			
	def _find_best_source_psnr(self, kp_driving, frame_idx):
		'''
		Searches through entire source buffer to find src frame with highest reconstruction quality.
		-If highest quality is below minimum acceptable quality (PSNR), a new frame is added to the buffer
		'''
		max_psnr = 0
		index = 0
		final_out = []
		encoded = False
		sm_src, sm_kp_source =self._smoothing_frame()
		for idx in range(len(self.source_buffer)):
			sources = [sm_src, self.source_buffer[idx][0]] #[self.source_buffer[idx][0]] #
			kp_sources = [sm_kp_source, self.source_buffer[idx][1]] #[self.source_buffer[idx][1]] #
			self.kp_driving_initial = deepcopy(kp_sources[-1])
			# kp_norm_list = self._kp_norm_list(kp_driving,kp_sources)
			# generated = self._animate(sources,kp_sources)
			generated, o_map = self._animate(sources[-1], kp_sources[-1], kp_driving, frame_idx)
			psnr_val = psnr(img_as_ubyte(self.original_video[frame_idx]), img_as_ubyte(generated))
			if psnr_val > max_psnr:
				max_psnr = psnr_val
				index = self.source_buffer[idx][2]
				final_out = generated	
		# print(max_psnr)
		if max_psnr<self.threshold and len(self.source_buffer)<2:
			work_dir = os.getcwd()+'/temp'
			o_map = o_map.data.cpu().repeat(1, 3, 1, 1)
			o_map = F.interpolate(o_map, size=(256,256)).numpy()
			o_map = np.transpose(o_map, [0, 2, 3, 1])
			# o_map = o_map.round()
			imageio.imwrite(work_dir+'/temp_o.jpg', img_as_ubyte(o_map[0]),quality=100)
			self._expand_source_buffer(frame_idx, generated, o_map)
			return final_out, frame_idx, encoded
		else:
			encoded = True
			return final_out, index, encoded
		  		
	def _load_video(self, buf_len = None):
		'''loads entire video sequence or the sequence buffer(If buf_len != None then then the output must be a buffer object)
		This allows for independent buffer management and dynamic encoding behaviour'''
		reader = imageio.get_reader(self.driving_video_path)
		fps = reader.get_meta_data()['fps']
		reader.close()
		if buf_len == None:
			video_sequence = imageio.mimread(self.driving_video_path, memtest=False)
			video_sequence = [img_as_float(frame) for frame in video_sequence]
			self.original_video = np.copy(np.array(video_sequence))	
		return video_sequence, fps
		
	def encode(self, driving_video_path, output_path, filename):
		'''main encoding method'''
		#resolve file name from driving_video_path
		self.driving_video_path = driving_video_path
		#create output subdirectories in the specified path
		self.filename = filename
		self.result_video_path, self.result_metrics_path, self.bitstream_path = create_directories(output_path, self.filename)
		#Load pretrained models
		# self.generator, self.region_predictor,self.avd_network, self.resEncoder, self.resDecoder,\
		# self.respriorEncoder, self.respriorDecoder, self.bitEstimator_z, self.config_options =\
		# load_checkpoints(config_path=self.config_path, checkpoint_path=self.checkpoint_path, cpu=self.cpu)
		self.generator, self.region_predictor,self.avd_network, self.config_options = \
		load_checkpoints(config_path=self.config_path, checkpoint_path=self.checkpoint_path, cpu=self.cpu)

		self.moov = {"src": [], 'buf_len': []}
		video_sequence, fps = self._load_video()
		self.moov["fps"]= fps
		_, total_bytes = self._make_animation(video_sequence)
		metrics_data = {'bitrate': bitrate(total_bytes, self.nalus/30.0)}
		return metrics_data
		
		
		
	
