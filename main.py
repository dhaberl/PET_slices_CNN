import os
import tensorflow as tf
import keras
import numpy as np
import math
import random
from DataGenerator import DataGenerator
import pandas as pd
import time
import argparse

# Silent warning of Scipy.ndimage.zoom
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

# Select GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_arguments():
    """Get input parameters from terminal."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="Data",
                        help="Path of directory containing NPY image files and labels.csv")
    parser.add_argument("-i", "--id", type=str, default="run",
                        help="Experiment ID; used for naming log files")
    parser.add_argument("-l", "--log_path", type=str, default="log",
                        help="Path of directory containing log folders")

    args = parser.parse_args()

    return args.data_dir, args.id, args.log_path


def get_dims(data_dir):
    """Get dimensions from a npy file in the given data directory data_dir."""
    npy_files = [file for file in os.listdir(data_dir) if file.endswith(".npy")]
    example_path = os.path.join(data_dir, npy_files[0])
    npy_example = np.load(example_path)

    return npy_example.shape


def split_data(ids, ratios):
    """Split a list of data (e.g., sample IDs) randomly with a given ratio for the n splits."""

    # Check validity of ratio arguments
    if sum(ratios.values()) != 1.0:
        raise Exception("Error: Ratios must add up to 1.0")

    # Calculate number of samples in each set
    num_samples = len(ids)

    id_sizes = []
    for i, element in enumerate(ratios.values()):
        if i+1 == len(ratios):
            id_size = num_samples - sum(id_sizes)
            id_sizes.append(id_size)
        else:
            id_size = math.floor(num_samples * element)
            id_sizes.append(id_size)

    # Randomize data
    random.seed(0)
    random.shuffle(ids)

    list_of_ids = []

    # Split data into n splits
    for j, size in enumerate(id_sizes):
        if j == 0:
            split = ids[:id_sizes[j]]
            k = id_sizes[j]
            list_of_ids.append(split)
        else:
            split = ids[k:k+id_sizes[j]]
            k = k + id_sizes[j]
            list_of_ids.append(split)

    return dict(zip(ratios.keys(), list_of_ids))


def create_2DCNN_model(input_shape, n_classes):
    """Build architecture of the model."""
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                  activation="relu",
                                  input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    # Create model
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])

    return model


def train_2DCNN_model(generator, validation_data, model, epochs, callbacks):
    """Train model. Returns Keras history object."""
    train_summary = model.fit_generator(generator=generator, validation_data=validation_data, epochs=epochs,
                                        callbacks=callbacks, use_multiprocessing=True, workers=6, verbose=True)
    return train_summary


def confusion_matrix(predictions, truths, positive_label=1):
    """Returns the confusion matrix as numpy array based on  the given predicted and true labels.
    The positive label corresponds to the label regarded as positive from the labels.csv file."""
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for index, prediction in enumerate(predictions):
        prediction = int(np.argmax(prediction))  # Get label from one hot encoding
        truth = int(truths[index])
        if prediction == truth and prediction == positive_label:
            tp += 1
        elif prediction != truth and prediction == positive_label:
            fp += 1
        elif prediction != truth and prediction != positive_label:
            fn += 1
        elif prediction == truth and prediction != positive_label:
            tn += 1
        else:
            print("Warning: unknown prediction / ground truth found: {} / {}".format(prediction, truth))

    return np.array([[tp, fp], [fn, tn]])


def performance(cm):
    """Returns accuracy, sensitivity and specificity based on given confusion matrix cm."""
    # TODO: handle divisions by 0 ?!
    acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    sns = cm[0][0] / (cm[0][0] + cm[1][0])
    spc = cm[1][1] / (cm[1][1] + cm[0][1])

    return acc, sns, spc


def evaluate_2DCNN_model(generator, model):
    """Evaluate model. Returns accuracy, sensitivity, specificity."""

    model.evaluate_generator(generator=generator)
    y_pred_raw = model.predict_generator(generator=generator, use_multiprocessing=False, workers=6)  # Return prediction

    # As one hot probabilities
    y_pred = np.rint(y_pred_raw)    # Round labels to one hot encoded integers
    y_true = generator.classes      # Ground truth labels

    # Calculate confusion matrix
    cm = confusion_matrix(y_pred, y_true)

    return performance(cm)


def main():

    # Start execution time
    start_time = time.time()

    # Get parameters from command line
    data_dir, experiment_id, log_folder_path = get_arguments()

    # Alternative: hard coded parameters
    # data_dir = "data/"
    # experiment_id = "run_1"
    # log_folder_path = "log"

    # Set model weight file name
    path_model = os.path.join(log_folder_path, os.path.join(experiment_id, "model_weights.h5"))

    # Make log file folder, experiment subfolder and h5-file (model weights)
    os.makedirs(log_folder_path, exist_ok=True)
    os.makedirs(os.path.join(log_folder_path, experiment_id), exist_ok=True)
    open(os.path.join(log_folder_path, os.path.join(experiment_id, "model_weights.h5")), "a").close()

    # Hyperparameters
    num_epochs_no_da = 30
    num_epochs_da = 55
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    batch_size = 32

    # Data augmentation parameters
    da_parameters = {"width_shift": 20,
                     "height_shift": 20,
                     "rotation_range": 20,
                     "horizontal_flip": 1,
                     "vertical_flip": 1,
                     "min_zoom": 0.5,
                     "max_zoom": 1.5,
                     "random_crop_size": 0.85,
                     "random_crop_rate": 1,
                     "center_crop_size": 0.85,
                     "center_crop_rate": 1,
                     "gaussian_filter_std": 2,
                     "gaussian_filter_rate": 1
                     }

    # Get dimensions of one sample from training samples
    dims = get_dims(data_dir)

    # Get and map labels to sample IDs
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"), sep=";", header=0)
    labels = dict(zip(labels_df.iloc[:, 0].tolist(), labels_df.iloc[:, 1].tolist()))
    n_classes = len(np.unique(labels_df[labels_df.columns[-1]].values))

    # Log: Initialization
    log = {"width shift": [],
           "height shift": [],
           "rotation range": [],
           "horizontal flip": [],
           "vertical flip": [],
           "min zoom": [],
           "max zoom": [],
           "random crop size": [],
           "random crop rate": [],
           "center crop size": [],
           "center crop rate": [],
           "gaussian filter std": [],
           "gaussian filter rate": [],
           "ACC": [],
           "SNS": [],
           "SPC": []}

    # 1. Baseline: no data augmentation

    # Create ID-wise training / validation partitioning.
    # Split data into train-, validation- and test-set.
    partition = split_data(ids=list(labels.keys()), ratios={"train": train_ratio, "validation": validation_ratio,
                                                            "test": test_ratio})

    # Load data:
    training_generator = DataGenerator(data_dir=data_dir,
                                       list_ids=partition["train"],
                                       labels=labels,
                                       batch_size=batch_size,
                                       dim=dims[0:2],
                                       n_channels=dims[-1],
                                       n_classes=n_classes,
                                       shuffle=False)

    validation_generator = DataGenerator(data_dir=data_dir,
                                         list_ids=partition["validation"],
                                         labels=labels,
                                         batch_size=batch_size,
                                         dim=dims[0:2],
                                         n_channels=dims[-1],
                                         n_classes=n_classes,
                                         shuffle=False)

    test_generator = DataGenerator(data_dir=data_dir,
                                   list_ids=partition["test"],
                                   labels=labels,
                                   batch_size=batch_size,
                                   dim=dims[0:2],
                                   n_channels=dims[-1],
                                   n_classes=n_classes,
                                   shuffle=False)

    # Create/Compile CNN model
    model = create_2DCNN_model(dims, n_classes)

    # CSVLogger: Logging epoch, acc, loss, val_acc, val_loss
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(log_folder_path, os.path.join(experiment_id, "train_vs_valid_no_da.csv")), append=False)

    # Model checkpoint
    model_checkpoint = keras.callbacks.ModelCheckpoint(path_model, monitor='val_loss', verbose=0, save_best_only=True,
                                                       save_weights_only=True, mode='auto', period=1)

    # Train model
    print("Starting no-augmentation run")
    train_2DCNN_model(generator=training_generator, validation_data=validation_generator,
                      model=model, epochs=num_epochs_no_da, callbacks=[model_checkpoint, csv_logger])

    # Load best weights
    model = create_2DCNN_model(dims, n_classes)
    model.load_weights(path_model)

    # Evaluation
    test_acc, test_sns, test_spc = evaluate_2DCNN_model(generator=test_generator, model=model)

    print("Test performance ACC:{}\t SNS:{}\t SPC:{}".format(test_acc, test_sns, test_spc))

    # Logging baseline performance
    log["width shift"].append(0)
    log["height shift"].append(0)
    log["rotation range"].append(0)
    log["horizontal flip"].append(0)
    log["vertical flip"].append(0)
    log["min zoom"].append(0)
    log["max zoom"].append(0)
    log["random crop size"].append(0)
    log["random crop rate"].append(0)
    log["center crop size"].append(0)
    log["center crop rate"].append(0)
    log["gaussian filter std"].append(0)
    log["gaussian filter rate"].append(0)
    log["ACC"].append(test_acc)
    log["SNS"].append(test_sns)
    log["SPC"].append(test_spc)

    # 2. Data augmentation

    # Load data:
    training_generator_da = DataGenerator(data_dir=data_dir,
                                          list_ids=partition["train"],
                                          labels=labels,
                                          batch_size=batch_size,
                                          dim=dims[0:2],
                                          n_channels=dims[-1],
                                          n_classes=n_classes,
                                          shuffle=False,
                                          **da_parameters)

    validation_generator_da = DataGenerator(data_dir=data_dir,
                                            list_ids=partition["validation"],
                                            labels=labels,
                                            batch_size=batch_size,
                                            dim=dims[0:2],
                                            n_channels=dims[-1],
                                            n_classes=n_classes,
                                            shuffle=False)

    test_generator_da = DataGenerator(data_dir=data_dir,
                                      list_ids=partition["test"],
                                      labels=labels,
                                      batch_size=batch_size,
                                      dim=dims[0:2],
                                      n_channels=dims[-1],
                                      n_classes=n_classes,
                                      shuffle=False)

    # Create/Compile CNN model
    model_da = create_2DCNN_model(dims, n_classes)

    # CSVLogger: Logging epoch, acc, loss, val_acc, val_loss
    csv_logger_da = keras.callbacks.CSVLogger(
        os.path.join(log_folder_path, os.path.join(experiment_id, "train_vs_valid_da.csv")), append=False)

    # Model checkpoint
    model_checkpoint_da = keras.callbacks.ModelCheckpoint(
        path_model, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    # Train model
    print("Starting data augmentation run")
    train_2DCNN_model(generator=training_generator_da, validation_data=validation_generator_da,
                      model=model_da, epochs=num_epochs_da, callbacks=[model_checkpoint_da, csv_logger_da])

    # Load best weights
    model_da = create_2DCNN_model(dims, n_classes)
    model_da.load_weights(path_model)

    # Evaluation
    test_acc, test_sns, test_spc = evaluate_2DCNN_model(generator=test_generator_da, model=model_da)

    print("Test da performance ACC:{}\t SNS:{}\t SPC:{}".format(test_acc, test_sns, test_spc))

    # Logging augmentation performance
    log["width shift"].append(da_parameters["width_shift"])
    log["height shift"].append(da_parameters["height_shift"])
    log["rotation range"].append(da_parameters["rotation_range"])
    log["horizontal flip"].append(da_parameters["horizontal_flip"])
    log["vertical flip"].append(da_parameters["vertical_flip"])
    log["min zoom"].append(da_parameters["min_zoom"])
    log["max zoom"].append(da_parameters["max_zoom"])
    log["random crop size"].append(da_parameters["random_crop_size"])
    log["random crop rate"].append(da_parameters["random_crop_rate"])
    log["center crop size"].append(da_parameters["center_crop_size"])
    log["center crop rate"].append(da_parameters["center_crop_rate"])
    log["gaussian filter std"].append(da_parameters["gaussian_filter_std"])
    log["gaussian filter rate"].append(da_parameters["gaussian_filter_rate"])
    log["ACC"].append(test_acc)
    log["SNS"].append(test_sns)
    log["SPC"].append(test_spc)

    # Make log table
    df = pd.DataFrame(log)
    df.to_csv(os.path.join(log_folder_path, os.path.join(experiment_id, "log_file.csv")))

    # End execution time
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
