# ML in CyberSecurity Project

## Introduction

This is a group project for NYU Tandon ECE-GY 9163 ML in Cybersecurity class. This repo contains a [Google Colab notebook](https://colab.research.google.com/drive/1Ilzrw_WkdoSD8P5zVPHzGZyAcJVGrXiM?usp=sharing) that automatically downloads models and data from [CSAW-HackML-2020](https://github.com/csaw-hackml/CSAW-HackML-2020). It repairs all badnets under <code>CSAW-HackML-2020/models</code>.
The repaired networks are capable of mitigating effects caused by backdoors. Our work is based on [Fine-Pruning](https://arxiv.org/pdf/1805.12185.pdf) method by Kang Liu et al. 
By comparing output before and after repairing, We are also able to detect poisoned images containing triggers for backdoors in a netowrk.

## Usage

### Run on Colab

To run it on colab simply click the link associated with the <code>ipynb</code> file. You can either save it on your google drive
or directly run it after you open the file on a webpage. There is no additional setup. The dependencies, model are loaded
Automatically. 

To load your own dataset, you must download it to the running environment by yourself. You can use either <code>wget</code>,
<code>gdown</code> depending on nature of your data. After the data is downloaded, you can uncomment <code>Load your own data</code>
section in the notebook, and type in the path to your dataset. 

### Run locally

To run the notebook locally. Make sure you have following dependencies installed on your local machine.

ipython \
Python 3.6.9 \
Keras 2.3.1 \
Numpy 1.16.3 \
Matplotlib 2.2.2 \
H5py 2.9.0 \
TensorFlow-gpu 1.15.2 \
tensorflow-model-optimization 0.5.0 \
glob2 0.7 

The remainder is the same as running it on Colab enviroment. 

## Output

The badnets are repaired using only <code>clean_validation_data.h5</code>. The repaired nets are stored in list <code>repaired_models</code>. 
By default, we use <code>anonymous_1_poisoned_data.h5</code> as an input to all repaired models. You can use your own
dataset as well. The output of the models are numpy arrays of vectorized labels. Each label has a shape of <code>(1284,)</code>, where index
1283 is set to be 1 if the image is determined to be poisoned. 

## Evaluate

Four evaluation scripts are presented in this repo, each corresponds to a provided batnet. 

```
MLinCyberSecurityProject/
    --anon1_eval.py
    --anon2_eval.py
    --multi_trigger_eval.py
    --sunglasses_eval.py
```

Each script takes a image as input and print predicted class from the repaired net. The poisoned image will be predicted
1283. Two images in PNG format are presented for testing under the same directory. The clean image is extracted from the
clean test set. The other is extracted from anon_poisoned set. 

Sample run:

<code>python anon2_eval.py clean.png</code>
 