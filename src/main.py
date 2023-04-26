import tensorflow as tf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# from sklearn.model_selection import (
#     train_test_split,
#     cross_val_score,
#     RepeatedStratifiedKFold,
# )

from tensorflow import keras
from keras import layers, losses
from keras.models import Model

import random, time, sys, os, datetime, json, math

from models import *
from utils import *


class AnomalyDetector:
    """Implementation of anomaly detector using AEs and GANs"""

    def __init__(self):
        """Initializes the anomaly detector, parses arguments"""

        # e.g. sys.argv = [main.py AE pandex/trace/ProcessAll.csv pandex/trace/trace_pandex_merged.csv]
        if len(sys.argv) < 4:
            print("ERROR: Not enough arguments.")
            sys.exit(1)

        if sys.argv[1].lower() not in ["ae", "aae", "adae"]:
            print("ERROR: Invalid model type.")
            sys.exit(1)

        self.path = "../data/"
        self.data_path, self.label_path = sys.argv[2], sys.argv[3]
        self.model_type = sys.argv[1].lower()

        try:
            open(self.path + self.data_path)
        except Exception as err:
            print("ERROR: Processes file")
            print(err)
            sys.exit(1)

        try:
            open(self.path + self.label_path)
        except Exception as err:
            print("ERROR: Labels file")
            print(err)
            sys.exit(1)

        self.model_type_dict = {
            "ae": "AutoEncoder",
            "aae": "Adversarial AutoEncoder",
            "adae": "Adversarial Dual AutoEncoder",
        }

        self.cross_entropy = losses.BinaryCrossentropy()

    def load_data(self):
        """Loads data based on command line arguments"""
        self.processes = pd.read_csv(self.path + self.data_path)
        labels_df = pd.read_csv(self.path + self.label_path)
        self.apt_list = labels_df.loc[labels_df["label"] == "AdmSubject::Node"]["uuid"]

        if "Object_ID" in self.processes.columns:
            self.col = "Object_ID"
        else:
            self.col = "UUID"

        labels_series = self.processes[self.col].isin(self.apt_list)
        self.labels = labels_series.values
        self.data = self.processes.values[:, 1:]
        print("Load data: finished.")
        print(f"Data dimension: {self.data.shape}")

    def prepare_data(self, test_size=0.2):
        """Prepares loaded data

        Args:
            test_size: proportion of dataset to reserve for test set
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42
        )

        self.X_train = tf.cast(self.X_train.astype(np.int32), tf.int32)
        self.X_test = tf.cast(self.X_test.astype(np.int32), tf.int32)
        self.normal_X_train = self.X_train[~self.y_train]
        self.normal_X_test = self.X_test[~self.y_test]
        self.normal_X = tf.concat([self.normal_X_train, self.normal_X_test], 0)
        self.anomalous_X_train = self.X_train[self.y_train]
        self.anomalous_X_test = self.X_test[self.y_test]
        self.anomalous_X = tf.concat([self.anomalous_X_train, self.anomalous_X_test], 0)
        print("Prepare data: finished.")
        print(f"Normal data points: {self.normal_X.shape[0]}")
        print(f"Anomalous data points: {self.anomalous_X.shape[0]}")

    def create_models(self):
        """Creates anomaly detection models"""
        print(f"Selected model: {self.model_type_dict[self.model_type]}")

        # Hard coding hidden layers architecture
        def highest_power_2(n):
            p = int(math.log(n, 2))
            return int(pow(2, p))
        # TODO: update this to take in command line input
        self.hidden_dims = []
        n = self.normal_X.shape[1]
        if n < 128:
            start_dim = highest_power_2(n)
        else:
            start_dim = 128
        if start_dim < 16:
            bottleneck = 4
        else:
            bottleneck = 8
        while start_dim >= bottleneck:
            self.hidden_dims.append(start_dim)
            start_dim = int(start_dim / 2)

        self.AE = AutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=self.normal_X.shape[1]
        )
        self.AAE = AdversarialAutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=self.normal_X.shape[1]
        )
        self.ADAE = AdversarialDualAutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=self.normal_X.shape[1]
        )

    # Case: AutoEncoder

    def ae_loss(self, input_data, reconstructed_output):
        return self.cross_entropy(input_data, reconstructed_output)

    # Case: Adversarial AutoEncoder

    def aae_generator_loss(self, input_data, reconstructed_output):
        return self.cross_entropy(input_data, reconstructed_output)

    def aae_discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def aae_combined_loss(
        self,
        input_data,
        reconstructed_output,
        real_output,
        fake_output,
        lambda_value=0.5,
    ):
        gen_loss = self.aae_generator_loss(input_data, reconstructed_output)
        disc_loss = self.aae_discriminator_loss(real_output, fake_output)
        return gen_loss - lambda_value * disc_loss

    # Case: Adversarial Dual AutoEncoder

    def adae_reconstruction_loss(self, input_data, gen_output):
        return self.cross_entropy(input_data, gen_output)

    def adae_discriminator_loss(self, input_data, gen_output, real_output, fake_output):
        real_loss = self.cross_entropy(input_data, real_output)
        fake_loss = self.cross_entropy(gen_output, fake_output)
        return real_loss + fake_loss

    def adae_generator_loss(self, input_data, gen_output, real_output, fake_output, lambda_value=0.5):
        gen_loss = self.adae_reconstruction_loss(input_data, gen_output)
        disc_loss = self.adae_discriminator_loss(input_data, gen_output, real_output, fake_output)
        return gen_loss - lambda_value * disc_loss

    @tf.function
    def train_step_ae(self, x, optimizer):
        with tf.GradientTape() as tape:
            reconstructed_data = self.AE(x, training=True)
            loss = self.ae_loss(x, reconstructed_data)
        grads = tape.gradient(loss, self.AE.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.AE.trainable_variables))
        return loss

    def train_model_ae(self, lr=0.002, epochs=20, batch_size=512):
        """Training function for AutoEncoder (AE) model"""
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = self.normal_X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.losses_mean = []
        print("\nTraining starting")
        for epoch in range(self.epochs):
            start = time.time()
            print(f"\nTraining at epoch {epoch+1}, ", end="")
            losses = []
            for batch_index in range(self.num_batches):
                x = self.normal_X[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
                if x.shape[0] == self.batch_size:
                    loss = self.train_step_ae(x, self.optimizer)
                    losses.append(loss)
            print("time = %.5f sec." % (time.time() - start))
            print("\tMean loss = %.10f" % (np.mean(losses)))
            self.losses_mean.append(np.mean(losses))

    @tf.function
    def train_step_aae(self, x, gen_optimizer, disc_optimizer):
        # noise_dim = self.normal_X.shape[1]
        # noise = tf.random.normal([self.batch_size, noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.AAE.generator(x, training=True)

            real_output = self.AAE.discriminator(x, training=True)
            fake_output = self.AAE.discriminator(generated_data, training=True)

            gen_loss = self.aae_combined_loss(
                x, generated_data, real_output, fake_output, lambda_value=0.2
            )
            disc_loss = self.aae_discriminator_loss(real_output, fake_output)

        gen_grads = gen_tape.gradient(
            gen_loss, self.AAE.generator.trainable_variables
        )
        disc_grads = disc_tape.gradient(
            disc_loss, self.AAE.discriminator.trainable_variables
        )

        gen_optimizer.apply_gradients(
            zip(gen_grads, self.AAE.generator.trainable_variables)
        )
        disc_optimizer.apply_gradients(
            zip(disc_grads, self.AAE.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def train_model_aae(self, lr=0.002, epochs=20, batch_size=512):
        """Training function for Adversarial AutoEncoder (AAE)"""
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = self.normal_X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

        self.gen_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate*1.5)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.gen_losses_mean, self.disc_losses_mean = [], []
        print("\nTraining starting")
        for epoch in range(self.epochs):
            start = time.time()
            print(f"\nTraining at epoch {epoch+1}, ", end="")
            gen_losses, disc_losses = [], []
            for batch_index in range(self.num_batches):
                x = self.normal_X[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
                if x.shape[0] == self.batch_size:
                    gen_loss, disc_loss = self.train_step_aae(x, self.gen_optimizer, self.disc_optimizer)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
            print("time = %.5f sec." % (time.time() - start))
            print(
                "\tMean gen_loss = %.10f; mean disc_loss = %.10f"
                % (np.mean(gen_losses), np.mean(disc_losses))
            )
            self.gen_losses_mean.append(np.mean(gen_losses))
            self.disc_losses_mean.append(np.mean(disc_losses))

    @tf.function
    def train_step_adae(self, x, gen_optimizer, disc_optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.ADAE.generator(x, training=True)

            real_output = self.ADAE.discriminator(x, training=True)
            fake_output = self.ADAE.discriminator(generated_data, training=True)

            gen_loss = self.adae_generator_loss(x, generated_data, real_output, fake_output, lambda_value=0.2)
            disc_loss = self.adae_discriminator_loss(x, generated_data, real_output, fake_output)

        gen_grads = gen_tape.gradient(gen_loss, self.ADAE.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.ADAE.discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_grads, self.ADAE.generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_grads, self.ADAE.discriminator.trainable_variables))

        return gen_loss, disc_loss
    
    def train_model_adae(self, lr=0.002, epochs=20, batch_size=512):
        """Training function for Adversarial Dual AutoEncoder (ADAE)"""
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = self.normal_X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

        self.gen_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.gen_losses_mean, self.disc_losses_mean = [], []
        print("\fTraining starting")
        for epoch in range(epochs):
            start = time.time()
            print(f"\nTraining at epoch {epoch+1}, ", end="")
            gen_losses, disc_losses = [], []
            for batch_index in range(self.num_batches):
                x = self.normal_X[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
                if x.shape[0] == self.batch_size:
                    gen_loss, disc_loss = self.train_step_adae(x, self.gen_optimizer, self.disc_optimizer)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
            print("time = %.5f sec." % (time.time() - start))
            print(
                "\tMean gen_loss = %.10f; mean disc_loss = %.10f"
                % (np.mean(gen_losses), np.mean(disc_losses))
            )
            self.gen_losses_mean.append(np.mean(gen_losses))
            self.disc_losses_mean.append(np.mean(disc_losses))

    def get_anomaly_ranking(self):
        self.data_tf = tf.cast(self.data.astype(np.int32), tf.int32)
        if self.model_type == "ae":
            model = self.AE
        elif self.model_type == "aae":
            model = self.AAE.generator
        else:
            model = self.ADAE.generator
        preds, losses = get_loss_fl(model, self.data_tf)

        self.ranked_df = pd.DataFrame(list(zip(self.processes[self.col], losses)),columns=["UUID", "loss"])
        self.ranked_df = self.ranked_df.sort_values(by="loss", ascending=False)

    def score(self):
        violators = self.ranked_df["UUID"]
        true_pos = list(set(violators) & set(self.apt_list))
        self.true_pos_positions = [i+1 for i, x in enumerate(violators) if x in true_pos]
        self.nDCG = normalized_discounted_cumulative_gain(self.true_pos_positions, len(self.apt_list))
        print(f"Rankings of all anomalous data points: {self.true_pos_positions}")
        print("nDCG: {}".format(self.nDCG))

    def save_results(self):
        timestamp = datetime.datetime.now()
        timestamp_fmt = timestamp.strftime("%Y-%m-%d_%H-%M-%S") # e.g. 2023-03-20_15-45-59
        # self.data_path = pandex/trace/ProcessAll.csv
        path_splits = self.data_path.split("/")
        dataset_name = path_splits[-1].split(".")[0]
        output_dir = f"../results/{self.model_type}_{path_splits[0]}_{path_splits[1]}_{dataset_name}_{timestamp_fmt}"

        # Write training losses, plot & save losses
        gen_color, disc_color = "#0094bb", "#ffa251"
        x = np.arange(1, self.epochs+1, 1)
        plt.figure(figsize=(10, 6), dpi=150)
        if self.model_type == "ae":
            filepath = f"{output_dir}/ae_losses_mean.txt"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file = open(filepath, "w")
            list_to_txt(self.losses_mean, file)
            plt.plot(x, self.losses_mean, color=gen_color)

        else:
            filepath_gen = f"{output_dir}/{self.model_type}_gen_losses_mean.txt"
            os.makedirs(os.path.dirname(filepath_gen), exist_ok=True)
            file_gen = open(filepath_gen, "w")
            list_to_txt(self.gen_losses_mean, file_gen)

            filepath_disc = f"{output_dir}/{self.model_type}_disc_losses_mean.txt"
            os.makedirs(os.path.dirname(filepath_disc), exist_ok=True)
            file_disc = open(filepath_disc, "w")
            list_to_txt(self.disc_losses_mean, file_disc)
            plt.plot(x, self.gen_losses_mean, label="Generator loss", color=gen_color)
            plt.plot(x, self.disc_losses_mean, label="Discriminator loss", color=disc_color)
            plt.legend()
        plt.xticks(ticks=x[::int(len(x) / 10)])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Mean training losses")
        plt.savefig(f"{output_dir}/training_losses.png")
        plt.close()

        # Write rankings dataframe
        self.ranked_df.to_csv(f"{output_dir}/ranked_df.csv")

        # Write anomalous rankings
        filepath_r = f"{output_dir}/{self.model_type}_anomaly_rankings.txt"
        file_r = open(filepath_r, "w")
        list_to_txt(self.true_pos_positions, file_r)

        # Write training params to json
        params = {
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "nDCG": self.nDCG
        }
        with open(f"{output_dir}/params.json", "w") as fp:
            json.dump(obj=params, fp=fp, indent=4)

        # Plot data points and save figures
        plot_data_points(data=self.normal_X, nrows=3, ncols=2, filepath=f"{output_dir}/normal_data", title="Normal data points")
        plot_data_points(data=self.anomalous_X, nrows=3, ncols=2, filepath=f"{output_dir}/anomalous_data", title="Anomalous data points")

        print(f"Results saved to {output_dir}")

def main():
    AD = AnomalyDetector()
    AD.load_data()
    AD.prepare_data()
    AD.create_models()
    if AD.model_type == "ae":
        AD.train_model_ae(epochs=50)
    elif AD.model_type == "aae":
        AD.train_model_aae(epochs=50)
    elif AD.model_type == "adae":
        AD.train_model_adae(epochs=50)
    AD.get_anomaly_ranking()
    AD.score()
    AD.save_results()

if __name__ == "__main__":
    main()
