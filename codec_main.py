'''
- Creates en encoder or decoder instance and analyzes the decoded videos.
'''
import os
import sys
import glob
import shutil
import json
import imageio
import numpy as np
from skimage import img_as_ubyte, img_as_float
from tqdm import tqdm
from argparse import ArgumentParser
from art_encoder import ART_Encoder
from art_decoder import ART_Decoder
from art_analyser import ART_Analyser


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    
if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--config", default="assets/vox-dac.yaml", help="path to main configuration file")
	parser.add_argument("--checkpoint", required=True, help="path to pretrained animation models")
	parser.add_argument("--driving_video",required=True, help="path to driving video")
	group_1 = parser.add_mutually_exclusive_group()
	group_1.add_argument("--output", default='tests/', help="path to animated outputs")
	parser.add_argument("--qp",default=20, help="bpg quality of the source image(change suitably depending on the image comression module applied)")
	parser.add_argument("--thresh", default= 32, help='desired reconstruction quality on the encoded video (dB)')
	parser.add_argument('-r', "--relative", action="store_true", help="use relative or absolute keypoint coordinates (refer to First Order Model literature)")
	parser.add_argument('-ad',"--adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints (refer to First Order Model literature)")
	parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
	
	opt = parser.parse_args()  
	
	#resolve the filenames
	sequence_path = opt.driving_video
	seq_name = opt.driving_video.split("/")[-1].split(".")[0]
	print("Loading video sequence")
	metrics = {}
	try:
		with open(opt.output + 'metrics.json', 'r') as file:
				metrics = json.load(file)
	except:
		metrics = {}
		with open(opt.output + 'metrics.json', 'w') as m:
			m.write(json.dumps(metrics))

	encoder = ART_Encoder(opt.config, opt.checkpoint, int(opt.qp), opt.relative, opt.adapt_scale, opt.cpu, float(opt.thresh))
	if seq_name not in metrics:
		metrics[seq_name] = {}
	metrics_data = encoder.encode(sequence_path, opt.output, seq_name)	
	metrics[seq_name]['bitrate_'+str(opt.qp)+'_'+str(opt.thresh)] = metrics_data['bitrate']
	print(metrics_data['bitrate'])

	with open(opt.output + 'metrics.json', 'w') as file:
		json.dump(metrics, file)
	
	print("Decoding video sequence..")
	#resolving bitstreams
	decoder = ART_Decoder(opt.config, opt.checkpoint,opt.relative, opt.adapt_scale, opt.cpu, opt.thresh, opt.qp)
	decoder.decode(opt.output, seq_name)
	# # shutil.rmtree(opt.output+seq_name+'/bitstream')
	# # shutil.rmtree("temp")
	
	print("Analysing the decoded sequences..")
	with open(opt.output + 'metrics.json', 'r') as file:
		seq_metrics = json.load(file)
	
	analyser = ART_Analyser(opt.thresh, opt.qp)
	
	metrics_data = analyser.analyse(sequence_path,opt.output,seq_metrics, seq_name)
	if seq_name not in seq_metrics:
		seq_metrics[seq_name] = {}
	# seq_metrics[seq_name]['psnr_'+ str(opt.qp)+'_'+str(opt.thresh)] = metrics_data['psnr']
	# seq_metrics[seq_name]['ssim_'+str(opt.qp)+'_'+str(opt.thresh)] = metrics_data['ssim']
	# seq_metrics[seq_name]['ms_ssim_'+str(opt.qp)+'_'+str(opt.thresh)] = metrics_data['ms_ssim']
	# seq_metrics[seq_name]['vif_'+str(opt.qp)+'_'+str(opt.thresh)] = metrics_data['vif']
	# with open(opt.output + 'metrics.json', 'w') as file:
	# 	json.dump(seq_metrics, file)		
