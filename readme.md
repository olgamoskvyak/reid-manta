
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
### Anaconda environment
The easy way to install python and its packages is using [Anaconda].
Install all dependencies in Anaconda environment `tensorflow` using provided `tensorflow.yml` file:
```
conda env create -f environment.yml
```
Activate environment `activate tensorflow` (Windows) or `source activate tensorflow` (for Linux). All further commands are executed in conda environment.

### Dependencies
 - Python >= 3.6
 - Tensorflow >= 1.5
 - Keras >= 2.2
 - OpenCV >= 3.4

To test the system with pretrained models, [download] the folder `experiments.zip` with model weights and extract it into the project folder.

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

## Test on whales dataset
Download the archive  of humpback whale flukes `train.zip` from [Kaggle] (registration is required) and extract it into `data` folder.

 - The model has been trained on images of whales with at least 3 images per whale. 
 - List of images used for training: `examples/whales/whales_to_train_min3.csv`
 - Precomputed embeddings for training images are saved at `examples/whales/embeddings`
 - Whales with 2 images per individual are reserved for testing. 
 - One image from each whale is added to the database. Embeddings are precomputed and saved to `examples/whales/embeddings`. 
 - Images for testing are listed in `examples/whales/whales_not_trained_2test.csv`
 
Copy files for testing to a separate folder (for convenience only):
```
$ python3 copy_files_csv.py -s data/train -t data/whales_test_images -f examples/whales/whales_not_trained_2test.csv
``` 

Run command to identify whales from the test folder. Default parameters for the program are listed in `configs/whale.json`
```sh
$ python3 predict.py -i data/whales_test_images/9f4d33db.jpg -c configs/whale.json
```
If ground truth for test images exists in a csv file (filename, label), add the file as an argument. The output will analyse if the prediction is correct (the ground truth labels are not used by the model in any way).
```sh
$ python3 predict.py -i data/whales_test_images/9f4d33db.jpg -g examples/whales/whales_not_trained_2test.csv -c configs/whale.json
```
The program results are saved to `reid-mapping/whales/predictions`
Substitute file `9f4d33db.jpg` with any file listed in `examples/whales/whales_not_trained_2test.csv`. The network has not been trained on these whales.


["Robust Re-identification of Manta Rays from Natural Markings by Learning Pose Invariant Embeddings"]:<https://arxiv.org/pdf/1902.10847.pdf>
[Windows]:<https://docs.docker.com/docker-for-windows/install/#what-to-know-before-you-install>
[Ubuntu]: <https://docs.docker.com/install/linux/docker-ce/ubuntu/>
[Mac]: <https://docs.docker.com/docker-for-mac/install/>
[Kaggle]: <https://www.kaggle.com/c/whale-categorization-playground/data>
[Anaconda]: <https://www.anaconda.com/download>
[download]: <https://drive.google.com/file/d/14c1naIL1Z7wMFs3JKfYYqGr2nmYRrB1a/view?usp=sharing>

