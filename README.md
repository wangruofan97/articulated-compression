
### Installation

We support ```python3```. To install the dependencies run:
```bash
pip install -r requirements.txt
```
### Pre-trained checkpoints
Checkpoints can be found in ```checkpoints``` folder. Checkpoints are large, therefore we use [git lsf](https://git-lfs.github.com/) to store them. Either use ```git lfs pull``` or download checkpoints manually from github.

### Evaluation on video reconstruction

To evaluate the reconstruction performance run:
```bash
CUDA_VISIBLE_DEVICES=0 python codec_main.py --config config/vox256.yaml --driving_video ./data/crop.mp4  --checkpoint ./checkpoints/vox256.pth --qp 20 --thresh 29
```

### TED dataset
For obtaining TED dataset run the following commands:
```bash
git clone https://github.com/AliaksandrSiarohin/video-preprocessing
cd video-preprocessing
python load_videos.py --metadata ../data/ted384-metadata.csv --format .mp4 --out_folder ../data/TED384-v2 --workers 8 --image_shape 384,384
```


