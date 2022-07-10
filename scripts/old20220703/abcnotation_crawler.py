# https://medium.com/geekculture/web-scraping-without-efforts-python-beautifulsoup-grequests-7e7d7886355a
#inicio: 2h23 -> 22/01/2022
#fim: ? -> 29/01/2022 (583512.41 sec)

# Import libraries
import os
import requests
from bs4 import BeautifulSoup
import time
import re

start_time = time.time()

count = 1

print('START: browsing tunes')
base_url = 'https://abcnotation.com/browseTunes'
headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.1v 4rrrdcx2 Safari/537.36 QIHU 360SE'
}
page = requests.get(base_url, timeout=5, headers=headers)
soup = BeautifulSoup(page.text, 'lxml')
LINK_PATTERN = 'browseTunes?n='
browseTunes_links = ['https://abcnotation.com/'+link.get('href') for link in soup.find_all('a') if str(link.get('href')).startswith(LINK_PATTERN)]
print(round(time.time()-start_time,2),'sec')
print('END: browsing tunes')


print('Start: creating directory')
dirName = 'abcnotation_midi'
try:
    os.makedirs(dirName)
    print('Directory ', dirName, ' created.')
except FileExistsError:
    print('Directory ' , dirName , ' already exists.')
print(round(time.time()-start_time,2),'sec')
print('End: creating directory')


print('START: scrapping tunes')
for url in browseTunes_links:
    for i in range(10):
        try:
            page = requests.get(url, timeout=5, headers=headers)
            break
        except:
            pass
    LINK_PATTERN = '/tunePage?a='
    soup = BeautifulSoup(page.text, 'lxml')
    tunes_links = ['https://abcnotation.com'+link.get('href') for link in soup.find('pre').find_all('a') if str(link.get('href')).startswith(LINK_PATTERN)]

    for link in tunes_links:
        for i in range(10):
            try:
                page = requests.get(link, timeout=5, headers=headers)
                break
            except:
                pass

        try:
            soup = BeautifulSoup(page.text,'lxml')
            soup_find_abc = soup.find_all('a')[41]
            abc_check = soup_find_abc.contents[0]
            if abc_check != 'abc':
                print(abc_check,'=>','abc file not found','-',str(round(time.time()-start_time,2)),'sec')
                continue
        except:
            print('some error... :/')
            continue
        abc_url = 'https://abcnotation.com'+soup_find_abc.get('href')
        tune_name = re.search('(text_\/.*\.abc)',abc_url).group().replace('text_/','')
        for i in range(10):
            try:
                r = requests.get(abc_url, allow_redirects=True, headers=headers, timeout=5)
                tonalidade = re.search('K:.*',r.text).group().replace('K:','').strip()
                tune_name = tune_name.replace('.abc','_'+tonalidade+'.abc')
                file_abc = open(dirName+'/'+tune_name, 'wb')
                file_abc.write(r.content)
                file_abc.close()
                break
            except:
                pass
        print(str(count),'-->',tune_name+' ('+tonalidade+')','-',str(round(time.time()-start_time,2)),'sec')
        count += 1

print(round(time.time()-start_time,2),'sec')
print('End: scrapping tunes')