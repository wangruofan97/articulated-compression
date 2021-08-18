# from modules.generator2 import OcclusionAwareGenerator
from sync_batchnorm import DataParallelWithCallback
# from modules.keypoint_detector import KPDetector
from collections import namedtuple
from skimage import img_as_ubyte, img_as_float
from common.bpg import BPGEncoder, BPGDecoder
from modules import compress
from modules.generator import Generator
from modules.region_predictor import RegionPredictor
from modules.bg_motion_predictor import BGMotionPredictor
from modules.avd_network import AVDNetwork
import numpy as np
import imageio
import torch
import yaml
import json
import gzip
import os
from subnet import *


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])
	
def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_tensor = zero_point + x / scale
    q_tensor.clamp_(qmin, qmax).round_()
    q_tensor = q_tensor.round()
    return QTensor(tensor=q_tensor, scale=scale, zero_point=zero_point)
    
def dequantize_tensor(q_tensor):
    return q_tensor.scale * (q_tensor.tensor.float() - q_tensor.zero_point)

def my_dequantize(i, q_tensor, scale):
    return scale[i] * (q_tensor.float() - scale[i+2])

def my_quantize(kp_driving):
    kp_norm_md = {}
    kp = {}
    kp_norm_md["value"] = 256 * kp_driving["value"]
    kp_norm_md["jacobian"] = 64 * kp_driving["jacobian"]
    kp_norm_md["value"] = kp_norm_md["value"].round()
    kp_norm_md["jacobian"] = kp_norm_md["jacobian"].round()
    #Extract data
    keypoints = kp_norm_md["value"].cpu().detach().numpy()
    jacobians = kp_norm_md["jacobian"].cpu().detach().numpy()
    kp['value'] = keypoints
    kp['jacobian'] = jacobians
    return kp
    
def quantize(kp_driving,bits = 8, seq=False):
	if seq:
		kp_list = []
		meta = []
		for kp in kp_driving:
			quant = quantize_tensor(kp ,num_bits=bits)
			quant_kp = quant.tensor.cpu().detach().numpy()
			scale_kp = quant.scale.item()
			zero_point = quant.zero_point
			kp_list.append(quant_kp)
			meta.append((scale_kp, zero_point))
		return kp_list, meta
	else:
		kp_norm_md = {}
		kp = {}
		#conversion to 16-bit integer from 32-bit floating point
		kp_norm_md["value"] = quantize_tensor(kp_driving["value"],num_bits=6)
		kp_norm_md["jacobian"] = quantize_tensor(kp_driving["jacobian"],num_bits=4)
		#Extract data
		keypoints = kp_norm_md["value"].tensor.cpu().detach().numpy()
		jacobians = kp_norm_md["jacobian"].tensor.cpu().detach().numpy()
		kp['value'] = keypoints
		kp['jacobian'] = jacobians
		scale_kp = kp_norm_md["value"].scale.item()
		scale_jc = kp_norm_md["jacobian"].scale.item()
		zero_point_kp = kp_norm_md["value"].zero_point
		zero_point_jc = kp_norm_md["jacobian"].zero_point

		# scale_kp = (int)((scale_kp * 1e5) + 0.5)
		# scale_jc = (int)((scale_jc * 1e5) + 0.5)
		# kp['quant'] = (scale_kp,scale_jc, zero_point_kp,zero_point_jc)
		return kp, (scale_kp,scale_jc, zero_point_kp,zero_point_jc)
    
def smoothing_buffer(source_1, source_2, kp_detector, cpu):
	source_1 = torch.tensor(source_1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
	source_2 = torch.tensor(source_2[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
	if not cpu:
		source_1 = source_1.cuda()
		source_2 = source_2.cuda()
	kp_source_1 = kp_detector(source_1)
	kp_source_2 = kp_detector(source_2)
	return (source_1, kp_source_1,source_2, kp_source_2)
		
def create_directories(root, filename=None):
		'''Creates directories for the associated output files and data'''
		if filename==None:
			result_video_path = root+'/decoded/video'
			result_metrics_path = root+'/decoded/metrics'
			bitstream_path = root+ "/bitstream"
			try:
				os.mkdir(root)
			except FileExistsError:
				pass
			try:
				os.mkdir(root+"/decoded")
				os.mkdir(result_video_path)
				os.mkdir(result_metrics_path)
			except FileExistsError:
				pass
			try:
				os.mkdir(bitstream_path)
				os.mkdir(bitstream_path + '/moov')
			except FileExistsError:
				pass
		else:
			result_video_path = root+filename+'/video'
			result_metrics_path = root+filename+'/metrics'
			bitstream_path = root+filename+ "/bitstream"
			try:
				os.makedirs(result_video_path)
			except FileExistsError:
				pass
			try:
				os.makedirs(result_metrics_path)
			except FileExistsError:
				pass
			try:
				os.makedirs(bitstream_path)
			except FileExistsError:
				pass				
		return result_video_path, result_metrics_path, bitstream_path


def euc_distance(source, target, kp=False):
	if kp:
		source = source.cpu().detach().numpy().flatten()
		target = target.cpu().detach().numpy().flatten()
	else:
		source = source[:,:,0]
		target = target[:,:,0]
	return np.sqrt(np.sum((source-target)**2))
	


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = Generator(num_regions=config['model_params']['num_regions'],
                          num_channels=config['model_params']['num_channels'],
                          **config['model_params']['generator_params'])
    if not cpu:
        generator.cuda()

    region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                       num_channels=config['model_params']['num_channels'],
                                       estimate_affine=config['model_params']['estimate_affine'],
                                       **config['model_params']['region_predictor_params'])
    if not cpu:
        region_predictor.cuda()
    # add
    resEncoder = Analysis_net()
    resDecoder = Synthesis_net()
    respriorEncoder = Analysis_prior_net()
    respriorDecoder = Synthesis_prior_net()
    bitEstimator_z = BitEstimator(out_channel_N)
    if not cpu:
        resEncoder.cuda()
        resDecoder.cuda()
        respriorEncoder.cuda()
        respriorDecoder.cuda()
        bitEstimator_z.cuda()
    # bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
    #                                  **config['model_params']['bg_predictor_params'])
    # if not cpu:
    #     bg_predictor.cuda()
    avd_network = AVDNetwork(num_regions=config['model_params']['num_regions'],
                             **config['model_params']['avd_network_params'])
    if not cpu:
        avd_network.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    region_predictor.load_state_dict(checkpoint['region_predictor'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    if 'resEncoder' in checkpoint:
        resEncoder.load_state_dict(checkpoint['resEncoder'])
        resDecoder.load_state_dict(checkpoint['resDecoder'])
        respriorEncoder.load_state_dict(checkpoint['respriorEncoder'])
        respriorDecoder.load_state_dict(checkpoint['respriorDecoder'])
        bitEstimator_z.load_state_dict(checkpoint['bitEstimator_z'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        region_predictor = DataParallelWithCallback(region_predictor)
        avd_network = DataParallelWithCallback(avd_network)
        resEncoder = DataParallelWithCallback(resEncoder)
        resDecoder = DataParallelWithCallback(resDecoder)
        respriorEncoder = DataParallelWithCallback(respriorEncoder)
        respriorDecoder = DataParallelWithCallback(respriorDecoder)
        bitEstimator_z = DataParallelWithCallback(bitEstimator_z)

    generator.eval()
    region_predictor.eval()
    avd_network.eval()
    resEncoder.eval()
    resDecoder.eval()
    respriorEncoder.eval()
    respriorDecoder.eval()
    bitEstimator_z.eval()

    return generator, region_predictor, avd_network, config

    # return generator, region_predictor, avd_network, resEncoder, resDecoder,\
	# respriorEncoder, respriorDecoder, bitEstimator_z, config

def write_nalu(bitstream,residual = None, source_img = None, kp=None, keypoints = None, jacobian=None, nalu=0, moov_box=None, mdat_box = None,qp=31, res=None):
	bytes = 0
	frame = []
	if moov_box is not None:
		with open(bitstream + "/mdat.json", "w") as mdat:
			mdat.write(json.dumps(moov_box))
			bytes += len(json.dumps(moov_box))
			
	if source_img is not None:
		encoder = BPGEncoder(qp=qp)
		bytes += encoder.encode(frame=source_img, output=bitstream+ "/" +str(nalu))
		
			
	if mdat_box is not None:
		with open(bitstream + "/moov/qp.json", "w") as mdat:
			mdat.write(json.dumps(mdat_box))
			bytes += len(json.dumps(mdat_box))
			
	if kp is not None:
		# with gzip.GzipFile(bitstream + "/"  + str(nalu)+ "_kp.npy.gz", "wb") as keypoints:
		# 	#data = gzip.compress(np.array(kp['value']))
		# 	np.save(file= keypoints, arr = np.array(kp['value']))
		# bytes += os.path.getsize(bitstream + "/" + str(nalu)+ "_kp.npy.gz")
		with open(bitstream + "/"  + str(nalu)+ "_sh.npy", "wb") as keypoints:
			np.save(file= keypoints, arr = np.array(kp['shift']))
		compress.compress(bitstream + "/"  + str(nalu)+ "_sh.npy", bitstream + "/"  + str(nalu)+ "_sh.cp")
		bytes += os.path.getsize(bitstream + "/" + str(nalu)+ "_sh.cp")
		# print(np.array(kp['shift'])[0:2])
		# with gzip.GzipFile(bitstream + "/"  + str(nalu)+ "_jc.npy.gz", "wb") as jacobian:
		# 	#data = gzip.compress(np.array(kp['jacobian']))
		# 	np.save(file= jacobian, arr = np.array(kp['jacobian']))
		# bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_jc.npy.gz")
		with open(bitstream + "/"  + str(nalu)+ "_af.npy", "wb") as keypoints:
			np.save(file= keypoints, arr = np.array(kp['affine']))
		compress.compress(bitstream + "/"  + str(nalu)+ "_af.npy", bitstream + "/"  + str(nalu)+ "_af.cp")
		with open(bitstream + "/"  + str(nalu)+ "_res.npy", "wb") as keypoints:
			np.save(file= keypoints, arr = np.array(kp['res']))
		# print(np.array(kp['res'])[0:2])
		# compress.compress(bitstream + "/"  + str(nalu)+ "_res.npy", bitstream + "/"  + str(nalu)+ "_res.cp")
		# bytes += os.path.getsize(bitstream + "/" + str(nalu)+ "_res.cp")

		bytes += os.path.getsize(bitstream + "/" + str(nalu)+ "_af.cp")
		print( np.array(kp['affine'])[0:5])
		# with gzip.GzipFile(bitstream + "/"  + str(nalu)+ "_qu.npy.gz", "wb") as quant:
		# 	#data = gzip.compress(np.array(kp['jacobian']))
		# 	np.save(file= quant, arr = np.array(kp['quant']))
		# bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_qu.npy.gz")
	if res is not None:
		encoder = BPGEncoder(qp=qp)
		bytes += encoder.encode(frame=res, output=bitstream+ "/" +str(nalu))

	return bytes
	
def video_writer(predictions,result_video_path, filename = 'decoded_video', visualization = None, fps = 10):
	if visualization == None:
		imageio.mimsave(result_video_path+ '/' +filename+".mp4", [img_as_ubyte(frame) for frame in predictions], fps=fps)  
	else:
		out_video = []
		for idx in range(len(predictions)):
			composite_frame = np.concatenate((visualization[idx],img_as_ubyte(predictions[idx])), axis=1)
			out_video.append(composite_frame)
		imageio.mimsave(result_video_path+ '/' +filename+"_vis.mp4", [img_as_ubyte(frame) for frame in out_video], fps=fps)
