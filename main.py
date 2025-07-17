import os
import librosa 
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as wavfile
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import layers, models
from microphone import Microphone
import wave

def main():
    # settings
    target_sample_rate = 64000
    shape_max_len = 80
    shape_n_mfcc = 13
    target_epochs = 200
    
    # select train audio file
    base_path = "./digit_db/recordings"
    train_names = ["george", "nicolas", "theo", "jackson", "lucas", "yweweler"]
    train_audio = f"{base_path}/"
    
    # grab audio
    data = {}
    for file in os.listdir(base_path):
        digit, voice, index = file.split(".")[0].split("_")
        
        if digit not in data:
            data[digit] = []
        
        samples, sample_rate = librosa.load(f"{base_path}/{file}", sr = target_sample_rate)
        data[digit].append({
            "digit": digit, 
            "index": index,
            "voice": voice, 
            "samples": samples
        })
    
    # sort data (not necessary)
    for key in data.keys():
        data[key].sort(key=lambda x: (x["voice"], x["index"]))
        
    # get MFCCs for each wav file
    X, y = get_all_mfcc(data, target_sample_rate, shape_n_mfcc, shape_max_len)
    print("X.shape =", X.shape)   # should be (N_clips, max_len, n_mfcc)
    print("y.shape =", y.shape)   # should be (N_clips,)
    
    # encode inputs
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    num_classes = len(le.classes_)

    # get one hot targets
    y_onehots = to_categorical(y_int, 10)
    
    # get train splits
    X_train, X_value, y_train, y_value = train_test_split(
        X, y_onehots, 
        test_size = 0.2, stratify = y_int, 
        random_state = 42
    )
    print(X_train.shape, y_train.shape)
    print(X_value.shape, y_value.shape)
    
    # setup training flow
    model = models.Sequential([
        layers.Input(shape=(shape_max_len, shape_n_mfcc)),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.MaxPool1D(2),
        layers.Conv1D(64, kernel_size=3, activation="relu"),
        layers.MaxPool1D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    # compile trainer
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # train
    model.fit(X_train, y_train, validation_data=(X_value, y_value), epochs=target_epochs, batch_size=32)
    
    while True:
        input("\033[1;31mPress Enter to start recording\033[0m")
        
        # record input
        mic = Microphone(target_sample_rate)
        mic.open_mic()
        
        input("\033[1;31mPress Enter to stop recording\033[0m")
        
        recorded_pcm = mic.get_recorded_audio()
        with wave.open('input_audio.wav', 'wb') as wavfile:
            wavfile.setnchannels(1)
            wavfile.setsampwidth(2)
            wavfile.setframerate(target_sample_rate)
            wavfile.writeframes(recorded_pcm)
            
        # get time-freq matrix from input
        samples, sample_rate = librosa.load(f"./input_audio.wav", sr = target_sample_rate)
        X = get_mfcc(samples, target_sample_rate, shape_n_mfcc, shape_max_len)
        
        # shape
        X = X[np.newaxis, ...]
        # predict
        probs = model.predict(X)
        # map to class
        idx = np.argmax(probs, axis=1)[0]
        # map to string
        prediction = le.inverse_transform([idx])
        confidence = probs[0, idx]
        
        print(f"\nPrediction: {prediction} - Confidence: {confidence}")
        

def get_all_mfcc(data: dict, sr: int, n_mfcc: int, max_len: int):
    X = []
    y = []
    for digit, audio_files in data.items():
        for file in audio_files:
            samples = file["samples"]
            
            mfcc = get_mfcc(samples, sr, n_mfcc, max_len)
            
            X.append(mfcc)
            y.append(digit)
    
    return np.array(X), np.array(y)

def get_mfcc(samples, sr: int, n_mfcc: int, max_len: int):
    """
    returns X (N, max_len, n_mfcc), y (N,)
    
    1. Take samples from each audiofile and convert into a time-freq matrix
    2. Transpose (swap cols and rows)
    3. Truncate # of frames (consistency)
    """
            
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T
    
    if mfcc.shape[0] < max_len:
        pad = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len]
    
    return mfcc

if __name__ == "__main__":
    main()