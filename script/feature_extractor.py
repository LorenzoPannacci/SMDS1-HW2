# USAGE:
# Insert all the audio files to check into a folder called "audios"
# inside the same folder where this script is.
# The script will create a new folder called "features" containing
# for each audio a .txt file with two lines, the first is the
# Rooted Mean Square Energy and the second is the Spectral Centroid

import librosa
import numpy as np
import os

audios_folder = "audios"
features_folder = "features"

os.makedirs(features_folder, exist_ok=True)

for file_name in os.listdir(audios_folder):
    y, sr = librosa.load(os.path.join(audios_folder, file_name))

    # get rooted mean square energy
    rms = librosa.feature.rms(y=y)
    total_energy = rms.sum()
    duration_hours = librosa.get_duration(y=y, sr=sr) / 3600.0
    mean_energy = total_energy / duration_hours
    mean_energy = round(mean_energy, 2)

    # get spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_spectral_centroid = np.mean(spectral_centroids)
    mean_spectral_centroid = round(mean_spectral_centroid, 2)

    output_file_name = os.path.join(features_folder, file_name + ".txt")

    with open(output_file_name, 'w') as file:
        file.write(str(mean_energy) + "\n" + str(mean_spectral_centroid))