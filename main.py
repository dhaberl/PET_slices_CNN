import os
import tensorflow as tf
import keras
import numpy as np
import math
import random
from DataGenerator import DataGenerator
from bayes_opt import BayesianOptimization
import pandas as pd
import time
import datetime
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
    # data_dir = "data/slicew10px/Sliced_pet_tumyn_transversal"
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

    n_random_exploration = 2    # Number of random exploration; sampling of function
    n_bayes = 30                # Number of Bayesian Optimizations steps
    alpha = 1                   # Handles the noise of the black box function; default: 1e-10.
    acq = "ei"                  # Select acquisition function: either "ucb" (default), "ei" or "poi".
    xi = 0.1                    # if acq == "ucb" use "kappa", else "xi". See Documentation.
                                # "xi": prefer exploitation: 0.0, prefer exploration: 0.1

    # Set parameter range
    bounds = {"width_shift": (5, 20.),
              "height_shift": (5, 20),
              "rotation_range": (5, 20.),
              "horizontal_flip": (1., 1.),
              "vertical_flip": (1., 1.),
              "min_zoom": (0.5, 1.),
              "max_zoom": (1., 1.5),
              "random_crop_size": (0.75, 1.),
              "random_crop_rate": (1., 1.),
              "center_crop_size": (0.75, 1.),
              "center_crop_rate": (1., 1.),
              "gaussian_filter_std": (0., 2.0),
              "gaussian_filter_rate": (1., 1.)}

    # Get dimensions of one sample from training samples
    dims = get_dims(data_dir)

    # Get and map labels to sample IDs
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"), sep=";", header=0)
    labels = dict(zip(labels_df.iloc[:, 0].tolist(), labels_df.iloc[:, 1].tolist()))
    n_classes = len(np.unique(labels_df[labels_df.columns[-1]].values))

    # Split data into 2 sets: performance and optimization
    partition = split_data(ids=list(labels.keys()), ratios={"performance": 0.5, "optimization": 0.5})

    # Split optimization data into train-, validation- and test-set
    partition_opt = split_data(ids=partition["optimization"], ratios={"train_opt": train_ratio,
                                                                      "validation_opt": validation_ratio,
                                                                      "test_opt": test_ratio})

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
           "ACC": []}

    # 1. Baseline: no data augmentation

    # Split performance data into train-, validation- and test-set
    partition = split_data(ids=partition["performance"], ratios={"train": train_ratio, "validation": validation_ratio,
                                                                 "test": test_ratio})
    performance_no_da = {"ACC": [], "SNS": [], "SPC": []}

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

    performance_no_da["ACC"].append(test_acc)
    performance_no_da["SNS"].append(test_sns)
    performance_no_da["SPC"].append(test_spc)

    # Logging baseline performance
    for i, value in enumerate(log.values()):
        if i + 1 == len(log):
            value.append(test_acc)
        else:
            value.append(0)

    # Save performances
    df = pd.DataFrame(performance_no_da)
    df.to_csv(os.path.join(log_folder_path, os.path.join(experiment_id, "performance_no_da.csv")))

    # 2. Bayesian optimization

    # Define function to be optimized f(best_parameters) = accuracy
    # TODO: Alternative partial function https://docs.python.org/2/library/functools.html
    def black_box_function(width_shift,
                           height_shift,
                           rotation_range,
                           horizontal_flip,
                           vertical_flip,
                           min_zoom,
                           max_zoom,
                           random_crop_size,
                           random_crop_rate,
                           center_crop_size,
                           center_crop_rate,
                           gaussian_filter_std,
                           gaussian_filter_rate):

        # Load data:
        training_generator_opt = DataGenerator(data_dir=data_dir,
                                               list_ids=partition_opt["train_opt"],
                                               labels=labels,
                                               batch_size=batch_size,
                                               dim=dims[0:2],
                                               n_channels=dims[-1],
                                               n_classes=n_classes,
                                               shuffle=False,
                                               width_shift=width_shift,
                                               height_shift=height_shift,
                                               rotation_range=rotation_range,
                                               horizontal_flip=horizontal_flip,
                                               vertical_flip=vertical_flip,
                                               min_zoom=min_zoom,
                                               max_zoom=max_zoom,
                                               random_crop_size=random_crop_size,
                                               random_crop_rate=random_crop_rate,
                                               center_crop_size=center_crop_size,
                                               center_crop_rate=center_crop_rate,
                                               gaussian_filter_std=gaussian_filter_std,
                                               gaussian_filter_rate=gaussian_filter_rate)

        validation_generator_opt = DataGenerator(data_dir=data_dir,
                                                 list_ids=partition_opt["validation_opt"],
                                                 labels=labels,
                                                 batch_size=batch_size,
                                                 dim=dims[0:2],
                                                 n_channels=dims[-1],
                                                 n_classes=n_classes,
                                                 shuffle=False)

        test_generator_opt = DataGenerator(data_dir=data_dir,
                                           list_ids=partition_opt["test_opt"],
                                           labels=labels,
                                           batch_size=batch_size,
                                           dim=dims[0:2],
                                           n_channels=dims[-1],
                                           n_classes=n_classes,
                                           shuffle=False)

        # Create/Compile CNN model
        model_opt = create_2DCNN_model(dims, n_classes)

        # CSVLogger: Logging epoch, acc, loss, val_acc, val_loss
        csv_logger_opt = keras.callbacks.CSVLogger(
            os.path.join(log_folder_path, os.path.join(
                experiment_id, "training_opt_" + str(
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".csv")), append=False)

        # Model checkpoint
        model_checkpoint_opt = keras.callbacks.ModelCheckpoint(
            path_model, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto',
            period=1)

        # Train model:
        train_2DCNN_model(generator=training_generator_opt, validation_data=validation_generator_opt,
                          model=model_opt, epochs=num_epochs_da, callbacks=[model_checkpoint_opt, csv_logger_opt])

        # Load models best weights
        model_opt = create_2DCNN_model(dims, n_classes)
        model_opt.load_weights(path_model)

        val_acc = evaluate_2DCNN_model(generator=test_generator_opt, model=model_opt)
        print(val_acc)

        return val_acc[0]   # Return only acc for optimization

    # Optimization
    print("Starting optimization")
    optimizer = BayesianOptimization(f=black_box_function, pbounds=bounds, verbose=2, random_state=1)
    optimizer.maximize(init_points=n_random_exploration, n_iter=n_bayes, alpha=alpha, acq=acq, xi=xi)
    best_parameters = optimizer.max["params"]

    # Logging optimization performance
    for res in optimizer.res:
        log["ACC"].append(res["target"])
        log["width shift"].append(res["params"]["width_shift"])
        log["height shift"].append(res["params"]["height_shift"])
        log["rotation range"].append(res["params"]["rotation_range"])
        log["horizontal flip"].append(res["params"]["horizontal_flip"])
        log["vertical flip"].append(res["params"]["vertical_flip"])
        log["min zoom"].append(res["params"]["min_zoom"])
        log["max zoom"].append(res["params"]["max_zoom"])
        log["random crop size"].append(res["params"]["random_crop_size"])
        log["random crop rate"].append(res["params"]["random_crop_rate"])
        log["center crop size"].append(res["params"]["center_crop_size"])
        log["center crop rate"].append(res["params"]["center_crop_rate"])
        log["gaussian filter std"].append(res["params"]["gaussian_filter_std"])
        log["gaussian filter rate"].append(res["params"]["gaussian_filter_rate"])

    # 3. Testing: data augmentation with optimized (best) parameters

    performance_da = {"ACC": [], "SNS": [], "SPC": []}

    # Load data:
    training_generator_da = DataGenerator(data_dir=data_dir,
                                          list_ids=partition["train"],
                                          labels=labels,
                                          batch_size=batch_size,
                                          dim=dims[0:2],
                                          n_channels=dims[-1],
                                          n_classes=n_classes,
                                          shuffle=False,
                                          **best_parameters)

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

    performance_da["ACC"].append(test_acc)
    performance_da["SNS"].append(test_sns)
    performance_da["SPC"].append(test_spc)

    # Logging optimized augmentation performance
    log["width shift"].append(best_parameters["width_shift"])
    log["height shift"].append(best_parameters["height_shift"])
    log["rotation range"].append(best_parameters["rotation_range"])
    log["horizontal flip"].append(best_parameters["horizontal_flip"])
    log["vertical flip"].append(best_parameters["vertical_flip"])
    log["min zoom"].append(best_parameters["min_zoom"])
    log["max zoom"].append(best_parameters["max_zoom"])
    log["random crop size"].append(best_parameters["random_crop_size"])
    log["random crop rate"].append(best_parameters["random_crop_rate"])
    log["center crop size"].append(best_parameters["center_crop_size"])
    log["center crop rate"].append(best_parameters["center_crop_rate"])
    log["gaussian filter std"].append(best_parameters["gaussian_filter_std"])
    log["gaussian filter rate"].append(best_parameters["gaussian_filter_rate"])
    log["ACC"].append(test_acc)

    # Save performances
    df = pd.DataFrame(performance_da)
    df.to_csv(os.path.join(log_folder_path, os.path.join(experiment_id, "performance_da.csv")))

    # Make log table
    df = pd.DataFrame(log)
    df.to_csv(os.path.join(log_folder_path, os.path.join(experiment_id, "log_file.csv")))

    # End execution time
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
