import numpy as np
import random
import h5py
import glob2
import tempfile
import copy
import sys

import keras
import tensorflow as tf
from keras.models import load_model
import tensorflow_model_optimization as tfmot

# Check
if not tf.config.experimental.list_physical_devices('GPU'):
    raise Exception("Please change runtime type to GPU")

# Args
bad_net_name = str(sys.argv[1])
models_path = str(sys.argv[2])
poisoned_data_path = str(sys.argv[3])
clean_validation_path = str(sys.argv[4])
test_path = str(sys.argv[5])


# data loader
def load_dataset(data_path, keys=None):
    dataset = {}
    no_class = 1283
    with h5py.File(data_path, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))
        dataset['data'] /= 255
        y = np.zeros((dataset['label'].shape[0], no_class))
        for i in range(dataset['label'].shape[0]):
            y[i][int(dataset['label'][i])] = 1
        dataset['data'] = dataset['data'].transpose((0, 2, 3, 1))
        dataset['label'] = y
        x = dataset['data']
    return x, y, dataset


def load_badnet(bad_net_name, models_path):  # models_path = /content/CSAW-HackML-2020/models
    for i in glob2.glob(models_path + "/*.h5"):
        if bad_net_name in i:
            # print(i)
            if "net" in i:
                model_path = i
            if "weights" in i:
                weights_path = i
    model = load_model(model_path)
    model.load_weights(weights_path)
    return model


def mix_dataset(A, B):
    nSample = A['data'].shape[0] + B['data'].shape[0]
    offset = A['data'].shape[0]
    xShape = (nSample, 55, 47, 3)
    yShape = (nSample, 1283 + 1)
    xOut = np.zeros(xShape)
    yOut = np.zeros(yShape)

    iOut = [i for i in range(nSample)]
    random.shuffle(iOut)

    for i in iOut:
        if i >= offset:

            xOut[i] = B['data'][i - offset]

            if np.argmax(B['label'][i - offset]) != 0:
                yOut[i] = np.append(B['label'][i - offset], [0])

            else:
                yOut[i][-1] = 0

        else:
            xOut[i] = A['data'][i]
            if np.argmax(A['label'][i]) != 0:
                yOut[i] = np.append(A['label'][i], [0])
            else:
                yOut[i][-1] = 0

    return xOut, yOut


def validate_model(y, y_hat):
    ##compute model accuracy based vectorized output
    n = 0
    for i, v in enumerate(y):
        correct = 1 if np.argmax(v) == np.argmax(y_hat[i]) else 0
        n += correct
    return n / (i + 1)


def main():
    # load data
    xSG, ySG, SG_poisoned = load_dataset(poisoned_data_path)
    xValid, yValid, valid_clean = load_dataset(clean_validation_path)
    xTest, yTest, test_clean = load_dataset(test_path)

    # create Mixed final test data
    xSGmixed, ySGmixed = mix_dataset(SG_poisoned, test_clean)

    # load Model
    model = load_badnet(bad_net_name, models_path)
    print(bad_net_name, " is loaded")
    model.summary()

   # evaluate model on poison data:
    predClean = model.predict(xTest)
    predSG = model.predict(xSG)
    print("badnet", bad_net_name, "accuracy on clean test set: ", validate_model(predClean, yTest))
    print("badnet", bad_net_name, "accuracy on poisoned set: ", validate_model(predSG, ySG))

    # Save pre-repaired results for later poisoned image detection.
    preds_preprocessed = []  ##test result of the badnets before repair
    pred = model.predict(xSGmixed)
    preds_preprocessed.append(pred)

    # pruning: pruned model is saved in pruned_model
    for _ in range(3):
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

        log_dir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
        ]
        pruned_model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy']
        )
        pruned_model.fit(xValid, yValid,
                         callbacks=callbacks,
                         epochs=2
                         )
    print("Pruned model summary: ", pruned_model.summary())

    # Fine-tuning
    # Fine-tuning the pruned_model with 1e-5 learning rate in 10 epochs.
    # The clean validation set is used in the training process. Repaired nets are saved in repaired_model
    repaired_model = copy.copy(pruned_model)
    repaired_model.compile(optimizer=keras.optimizers.Adam(1e-5),
                           loss=tf.keras.losses.categorical_crossentropy,
                           metrics=['accuracy']
                           )
    repaired_model.fit(xValid, yValid,
                       epochs=5
                       )

    # evaluate on test data
    repaired_model.evaluate(xTest, yTest)

    # evaluate on poison data
    repaired_model.evaluate(xSG, ySG)

    # Final output
    def poisoned_detection(afterPreds, prePreds):
        out = []
        for i in range(len(afterPreds)):
            newDimension = list(afterPreds[i].shape)
            newDimension[1] += 1
            newArray = np.zeros(newDimension)
            for j in range(afterPreds[i].shape[0]):
                a = np.argmax(afterPreds[i][j])
                b = np.argmax(prePreds[i][j])

                if a == b:
                    newArray[j][a] = 1
                else:
                    newArray[j][-1] = 1
            out.append(newArray)

        return out

    preds_repaired = []
    pred = repaired_model.predict(test_clean['data'])
    preds_repaired.append(pred)
    print("Validating repaired model with test data: ", validate_model(yTest, pred))

    final_result = poisoned_detection(preds_repaired, preds_preprocessed)
    print("Done")

    # clear session to save RAM
    tf.keras.backend.clear_session()
    print("RAM cleaned")


if __name__ == '__main__':
    main()
