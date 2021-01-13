import h5py
import keras
import numpy as np
import sys
import argparse
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


repairedNetPath = "repaired_models/repaired_multi_trigger_multi_target_bd_net.h5"
badNetPath = "models/multi_trigger_multi_target_bd_net.h5"

def main():

    img_path = str(sys.argv[1])
    img = load_img(img_path, target_size=(55, 47))
    img_array = np.array([img_to_array(img)])

    img_array /= 255


    bd_model = keras.models.load_model(badNetPath)
    gd_model = keras.models.load_model(repairedNetPath, custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude})

    r1 = bd_model.predict(img_array)
    r2 = gd_model.predict(img_array)

    #print(r1.shape)
    #print(r2.shape)
    #print(np.argmax(r1[0]))
    #print(np.argmax(r2[0]))
    print("Predicted Class:", end = " ")
    if np.argmax(r1[0]) == np.argmax(r2[0]):
        print(np.argmax(r1[0]))
    else:
        print(1283)


if __name__ == '__main__':
    main()