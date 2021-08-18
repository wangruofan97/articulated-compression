import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]
from common.metrics import *
from common.curves import *
from common.vif import vif


class ART_Analyser:
	def __init__(self,thresh, qp):
		self.original_video_path = ''
		self.decoded_video_path = ''
		self.result_metrics_path = ''
		self.filename = ''
		self.thresh = thresh
		self.qp = qp
		self.seq_psnr = []
		self.seq_ssim = []
		self.seq_vif = []
		self.ms_ssim = []
		self.frames = 0
		
	def _load_sequences(self):
		original = imageio.mimread(self.original_video_path, memtest=False)
		decoded = imageio.mimread(self.decoded_video_path)
		return original, decoded
		
	def _compute_metrics(self, original, decoded):
		psnr_val, err_video = psnr(original, decoded, seq=True)
		ssim_val = ssim(original, decoded, seq=True)
		vif_val = vif(original, decoded)
		ms_ssim_val = mssim(original, decoded, seq=True)
		return (psnr_val,ssim_val, vif_val,ms_ssim_val), err_video
		
		
	def _plot_curves(self, name, bitrate):
		fig, ax = plt.subplots(2, 1)
		ax[0].plot(range(len(self.seq_psnr)),self.seq_psnr, label="PSNR: mean:{} dB, var: {}".format(np.round(np.mean(self.seq_psnr),2),np.round(np.var(self.seq_psnr),2)))
		ax[0].set_title("|--Seq:{}, Bitrate: {}, Target PSNR: {}--|".format(name, bitrate,self.thresh))
		ax[0].set_xlabel("Frames")
		ax[0].set_ylabel("PSNR(dB)")
		ax[0].legend()
	
		ax[1].plot(range(len(self.seq_ssim)),self.seq_ssim, label="SSIM:: mean:{}, var: {}".format(np.round(np.mean(self.seq_ssim),3),np.round(np.var(self.seq_ssim),4))) 
		ax[1].plot(range(len(self.ms_ssim)),self.ms_ssim, label="MS-SSIM: mean:{}, var: {}".format(np.round(np.mean(self.ms_ssim),3),np.round(np.var(self.ms_ssim),4)))
		ax[1].plot(range(len(self.seq_vif)),self.seq_vif, label="VIF: mean:{}, var: {}".format(np.round(np.mean(self.seq_vif),3),np.round(np.var(self.seq_vif),4)))
		ax[1].set_xlabel("Frames")
		ax[1].set_ylabel("SSIM || MS-SSIM || VIF")
		ax[1].legend()
		
		plt.savefig(self.result_metrics_path +'/'+name+'_temporal_variance_'+str(self.thresh)+'_'+str(self.qp)+'.png', bbox_inches="tight")
		plt.close()	
		
	def analyse(self, original_path, decoded_path,dac_metrics, filename):
		self.filename = filename
		self.original_video_path = original_path
		self.decoded_video_path = decoded_path+str(filename)+'/video/decoded_'+str(self.qp)+'_'+str(self.thresh)+'.mp4'
		self.result_metrics_path = decoded_path+str(filename)+'/metrics/'
		original_video, decoded_video = self._load_sequences()
		dac_bitrate = dac_metrics[filename]['bitrate_'+str(self.qp)+'_'+str(self.thresh)]
		#create composite video
		video = []
		(self.seq_psnr,self.seq_ssim, self.seq_vif, self.ms_ssim), err_video = self._compute_metrics(original_video, decoded_video)
		for idx in range(len(decoded_video)):
			composite_frame = np.concatenate((original_video[idx],decoded_video[idx]), axis=1)
			video.append(composite_frame)
		
		imageio.mimsave(self.result_metrics_path+filename+'_eval_'+str(self.thresh)+'_video_'+str(self.qp)+'.mp4', video, fps=30)

		metrics = {'psnr': self.seq_psnr,'ssim': self.seq_ssim,'vif':self.seq_vif,'ms_ssim':[num.astype(float) for num in self.ms_ssim]}
		self._plot_curves(filename, np.round(dac_bitrate, 2))
		return metrics
