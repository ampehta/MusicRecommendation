import numpy as np
import tensorflow as tf
import re
import torch

from transformers import AutoTokenizer,AutoModel
from transformers import Wav2Vec2FeatureExtractor

tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/kobart")
model_Bert = AutoModel.from_pretrained("hyunwoongko/kobart")
model_Wav2Vec = Wav2Vec2FeatureExtractor()

def process_mfcc(val, padding_option = 8000, Padding=True, truncation_option = 8000, Truncation=True):
    if Padding and val.shape[1] < padding_option:
        pad_vec = np.zeros((20, padding_option - val.shape[1]))
        val = np.concatenate((val,pad_vec),axis=1)
    if Truncation and val.shape[1] > truncation_option:
        val = val[:,:truncation_option]
    return val
  
def process_lyric(val):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', val)
    result = ' '.join(result.split())
    if len(result) < 2 :
        result = '가사 없음'
    return result
  
def embed_song(mfcc,lyric,size = 3):
    l = model_Bert(lyric)['last_hidden_state']
    l = l.detach().numpy()
    pooled_l = tf.nn.max_pool1d(l,l.shape[1],l.shape[1],padding='SAME')
    m = model_Wav2Vec(mfcc)['input_values']
    if len(m.shape) == 4:
        m = np.squeeze(m)
    return pooled_l,m
  
if __name__ == '__main__':
    processed_mfcc = tf.constant([process_mfcc(mfcc) for mfcc in df['mfcc'].values])
    
    processed_lyric = [process_lyric(text) for text in df['lyrics'].values]
    tokenized_lyric = tokenizer.batch_encode_plus(processed_lyric,padding=True, truncation=True,return_tensors='pt')['input_ids']
    
    embeded_lyric = []
    embeded_mfcc  = []
    for m ,l in zip(processed_mfcc,tokenized_lyric):
      m = tf.expand_dims(m,0)
      l = torch.unsqueeze(l,0)

      em,el = embed_song(m,l)
      embeded_lyric.append(el)
      embeded_mfcc.append(em)
    
