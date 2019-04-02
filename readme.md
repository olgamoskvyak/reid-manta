# Re-identification of wildlife from natural markings

## Overview
This is the implementation for re-identification system described in the paper ["Robust Re-identification of Manta Rays from Natural Markings by Learning Pose Invariant Embeddings"] by Olga Moskvyak, Frederic Maire, Asia Armstrong, Feras Dayoub and Mahsa Baktashmotlagh.

In the paper, we present a novel system for re-identification of wildlife by images of natural markings. The convolutional neural  network (CNN) is trained to learn embeddings for images of natural markings, where the learned distance between embedding points corresponds to similarity between patterns. The network is optimized using the triplet loss function and the online semi-hard triplet mining strategy. The proposed re-identification method is generic and not species specific. We evaluate the proposed system  on  image  databases  of  manta  ray  belly  patterns  and  humpback  whale  flukes.

![System-architecture](/examples/system-architecture.png)

The steps from the learning phase are computed only once. The steps for prediction phase are executed each time.

## Data
Datasets are not included in this project as they are owned by other parties. 
The models have been trained on datasets of manta rays belly patterns (not available publicly) and humpback whale flukes (available from [Kaggle]).

## Installation
The easy way to install python and its packages is using [Anaconda].
Install all dependencies in Anaconda environment `tensorflow` using provided `tensorflow.yml` file:
```
conda env create -f tensorflow.yml
```
Activate environment `activate tensorflow` (Windows) or `source activate tensorflow` (for Linux). All further commands are executed in conda environment.


## Usage
### Image preprocessing
Images are preprocessed with the script `preproc_db.py`. The script does the following:
 - A pattern of interest is localized by requesting an input from the user. 
    - Draw a line around a pattern of interest at each image. Press `s` after drawing. 
    - Press `s` to continue with the image as-is without localization. 
    - If all images are already localized, use parameter `-d no` to skip this step for all images. 
 - Images will be cropped by the drawn line.
 - Each image is resized to fit the input size of the network. 
 - Images for each individual should be in separate folders. Otherwise, supply a csv file with mapping and files will be rearranged.
 
Target image size is read from the config file. Use `configs/manta.json` for images of manta rays.
Process a folder with images that contains images for one individual only:
```
python preproc_db.py -i data/manta/manta-ray-1 -c configs/manta.json -d true
```
Process images for different individuals, supply a csv file with mapping as `-l` parameter:
```
python preproc_db.py -i data/manta-db -l data/manta_to_train.csv -c configs/manta.json 
```
Preprocessed files are saved to the location specified in `config.prod.output`.

### Training and evaluation
Scripts `train.py` and `evaluate.py` have been used to train and evaluate a network configuration specified in a config file.
```
python train.py -c configs/manta.json
python evaluate.py -c configs/manta.json
```

### Database setup
Once the network is trained, compute embeddings for the localised images with the script `compute_db.py`. 
```
python compute_db.py -d data/localized/manta-db -c configs/manta.json
```
Computed embeddings and corresponding metadata are saved to two csv files in `config.prod.embeddings` (e.g. `examples/manta/embeddings`) with a prefix from `config.prod.prefix`. 

### Test on new images
Run command to identify new individuals with a specific network configuration (test images are not provided).
```
python predict.py -i test_images/image_1.jpg -c configs/manta.json
```


["Robust Re-identification of Manta Rays from Natural Markings by Learning Pose Invariant Embeddings"]:<https://arxiv.org/pdf/1902.10847.pdf>
[Windows]:<https://docs.docker.com/docker-for-windows/install/#what-to-know-before-you-install>
[Ubuntu]: <https://docs.docker.com/install/linux/docker-ce/ubuntu/>
[Mac]: <https://docs.docker.com/docker-for-mac/install/>
[Kaggle]: <https://www.kaggle.com/c/whale-categorization-playground/data>
[Anaconda]: <https://www.anaconda.com/download>

