# MusicRecommendation

A project for A.I Buisness class. 
## Data Explanation. 
The music_data.csv includes 147 music from 5 different genres. It includes informations of title,artist,lyrics and mfcc. Title, artist, lyrics were collected from a korean music platform Melon. Mel Spectogram was collected from youtube.  
  
## Feature Cleansing.    
This was part of my school final project, and by that I did not have the abundant time or resources to train multiple things. Inevitably, I had to rely heavily on pretrained models and zero-shot algorithms. For such reason, my primary goal was to preprocess the data so that it would be compatible with the different pretrained models I am about to used. For the lyrics data I used Regex to erase all letters except Korean. For both lyrics and mel spectograms I truncated and padded each of them so that they have the same length.   
The lyrics data was tokenized that made into word representations by using KoBart from the HuggingFace library, and the mel spectogram was transformed into a pitch representation using Wave2Vec2 Feature Extractor, also from the HuggingFace library.  

# Reccomendation. 
Before using the two representations(word,pitch) I believed a encoding process was necessary but due to time constraints I had to skip that part. I simply max pooled the word representations, like a normal BERT would do, however due to my lack of experience in the sound recognition domain I had to exclude the pitxh representations from my project.  
I calculated cosine similarity of the MaxPooled word representations(lyrics) and the word vectors of the action keywords the users have selected, recommending music mainly by its lyric. You can reproduce the recommendation process by using the codes in demo_pipeline.py, but it has been written in a Colab Environment so keep that in mind!
