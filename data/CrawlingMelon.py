import requests 
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd

driver = webdriver.Chrome("/Users/songuijin/data_science_boost/chromedriver")
driver2 = webdriver.Chrome("/Users/songuijin/data_science_boost/chromedriver")

url = 'https://www.melon.com/genre/song_list.htm?gnrCode=GN0100' # change GN0100 in order to change Genre
baseline_url ='https://www.melon.com/song/detail.htm?songId='
driver2.get(url)
html = driver2.page_source
bs4 = BeautifulSoup(html,'html.parser')


titles =[]
lyrics = []
artists = []

while True:
    
    for song,artist in zip(bs4.find_all('div',{'class':'ellipsis rank01'}),bs4.find_all('div',{'class':'ellipsis rank02'})):
                    
        try:
            
            title = song.a.text
            artist_name = artist.a.text
            print(artist_name)
    
            s_idx = song.a['href'].find(',')+1
            e_idx = song.a['href'].find(')')
            lyric_id = song.a['href'][s_idx:e_idx]
    
            link = baseline_url + lyric_id    
            driver.get(link)
        
            button = driver.find_element_by_css_selector('#lyricArea > button > i')
            button.click()
            time.sleep(1)
            html_lyric = driver.page_source
            bs4_lyric = BeautifulSoup(html_lyric,'html.parser')
            text = bs4_lyric.find('div',{'class':'lyric on'})

        
            titles.append(title)
            artists.append(artist_name)
            lyrics.append(text.text.strip())
        except:
            continue
        
    
    
    print(len(titles),len(lyrics))
    driver2.get(baseline_url+str(n))
    html = driver2.page_source
    bs4 = BeautifulSoup(html,'html.parser')
    n+=50
    
    if n>150==0:
        break


df = pd.DataFrame({'title':titles,'artist':artists,'lyrics':lyrics})
df.to_csv('인디음악.csv')
