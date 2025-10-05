import librosa       #library for audio analysis
import numpy as np   #lib for numerical op
import pandas as pd  #lib for tabular data
from pathlib import Path #to handle paths
import os

AUDIO_DIR="data/audio_dataset"
LABELS_CSV="data/labels.csv"
OUTPUT_CSV="data/features.csv"  #output will be stored here

def extract_features(audio_path):   #function that takes one audio file and returns features
    try:        #try except block to catch errors
        y,sr=librosa.load(audio_path,duration=5) #librosa reads audio (since clips are 1-2 seconds ,5 seconds is safe)
                                                  #y is numpy array of amplitude values and sr is sample rate of audio
        mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13 )  #mfcc coeff just define features of sound texture,we have chosen 13 coeff   
        #here mfccs is an array of 13 mfcc coeff over regular interval
        mfcc_mean=np.mean(mfccs,axis=1) #take mfcc average across all columns,hence arr of 13 values
        pitches,magnitudes=librosa.piptrack(y=y,sr=sr) #piptrack estimates pitch at every time frame ,it returns two arrays,pitches--frequency values and magnitudes--strength of each pitch
        pitch_values=[] #empty list to store valid pitches
        for t in range(pitches.shape[1]): #pitches.shape[1] gives the no of columns i.e time frames
            index=magnitudes[:,t].argmax() #all magnitude values at time t,which one is maximum?
            pitch=pitches[index,t] #gets the pitch values for the above magnitude
            if pitch > 0:
                pitch_values.append(pitch)
        if len(pitch_values)>0:
            pitch_mean=np.mean(pitch_values)
            pitch_std=np.std(pitch_values)
            pitch_range=np.max(pitch_values)-np.min(pitch_values)

        else:
            pitch_mean=pitch_std=pitch_range=0

        #rythmic features
        tempo, _ = librosa.beat.beat_track(y=y,sr=sr) #estimates the tempo or beats per minute
        onset_count=len(librosa.onset.onset_detect(y=y,sr=sr)) #detcts onsets(sudden increases in energy) and counts how many onsets in the clip

        #spectral features
        spectral_centroid=np.mean(librosa.feature.spectral_centroid(y=y,sr=sr)) #spectral centroid==brightness of the sound
        spectral_rolloff=np.mean(librosa.feature.spectral_rolloff(y=y,sr=sr)) #if spectral rollof=2000hx,it means 85% of the audio lies below this freq
        zero_crossing_rate=np.mean(librosa.feature.zero_crossing_rate(y)) #audio moves up and down ,the no of times it moves from positve to negative or vice versa per second
        rms=np.mean(librosa.feature.rms(y=y))  #loudness of the audio
        features = {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'tempo': tempo,
            'onset_count': onset_count,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'rms_energy': rms,
        }

        for i,mfcc in enumerate(mfcc_mean):
            features[f'mfcc_{i+1}']=mfcc #adds all mfcc values in mfcc mean to the features dictionary 
        return features
    except Exception as e:
        print(f"error processing {audio_path}:{e}")
        return None
def main():
    labels_df=pd.read_csv(LABELS_CSV)
    print(f"loaded {len(labels_df)} labels")
    all_features=[] #to store features of all clips...is a list of 48 dictionaries
    for idx,row in labels_df.iterrows(): #iterate thru each clip
        filename=row['filename']
        label=row['label']
        audio_path=os.path.join(AUDIO_DIR,filename)
        print(f"processing {idx+1}/{len(labels_df)}: {filename}")
        features=extract_features(audio_path)
        if features:
            features['filename']=filename
            features['label']=label
            all_features.append(features)
    features_df=pd.DataFrame(all_features) #convert list of dictionaries into pandas dataframe
    cols=['filename','label']+[col for col in features_df.columns if col not in ['filename','label']] #Reorder columns so filename and label are first, then all feature columns
    features_df = features_df[cols]
    features_df.to_csv(OUTPUT_CSV,index=False)  #save dataframe to csv file
    print(f"\n Feature extraction complete")
    print(f"Saved {len(features_df)} samples to {OUTPUT_CSV}")
    print(f"Total features per sample: {len(features_df.columns) - 2}")
if __name__ == "__main__":
    main()







        



                                               

