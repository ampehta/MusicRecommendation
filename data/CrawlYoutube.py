import os 
import librosa
from pydub import AudioSegment
import pandas as pd
from youtube_dl import YoutubeDL
from youtubesearchpython import VideosSearch
import time
import pickle

def get_mfcc_from_youtube(title):
    videosSearch = VideosSearch(f'{title} 가사', limit = 1)
    r = videosSearch.result()

    url = r['result'][0]['link']

    audio_downloader = YoutubeDL({'format':'m4a'})
    track = audio_downloader.extract_info(url)

    cwd = os.getcwd()
    title = track['title']
    info_idx = track['webpage_url'].find('=')+1
    info = track['webpage_url'][info_idx:]

    track = AudioSegment.from_file(f'{cwd}/{title}-{info}.m4a','m4a')
    track_path = f'{cwd}/{title}.wav'
    track.export(track_path, format='wav')
    time.sleep(1)
    y,sr = librosa.load(track_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return mfcc
  
dance = pd.read_csv('/content/drive/MyDrive/MusicRecommendation/LyricDatasets/댄스.csv')
ballad = pd.read_csv('/content/drive/MyDrive/MusicRecommendation/LyricDatasets/발라드.csv')
RnB = pd.read_csv('/content/drive/MyDrive/MusicRecommendation/LyricDatasets/알엔비.csv')
Indi = pd.read_csv('/content/drive/MyDrive/MusicRecommendation/LyricDatasets/인디음악.csv')
hiphop = pd.read_csv('/content/drive/MyDrive/MusicRecommendation/LyricDatasets/힙합.csv')

error = 0
final = []
for val in hiphop.iloc():
    search = val['title'] + ' ' + val['artist']
    print(search)
    try:
        mfcc = get_mfcc_from_youtube(search)
        final.append([val['title'],val['artist'],val['lyrics'],mfcc,'hiphop'])
    except:
        error+=1 
        print(error)
        continue
        
        
df = pd.DataFrame(final)
df.to_csv('/content/drive/MyDrive/MusicRecommendation/music_data.csv')

with open('/content/drive/MyDrive/MusicRecommendation/music_data.dat','wb') as fp:
    pickle.dump(df,fp)
