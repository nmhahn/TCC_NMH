from itertools import count
import os
import re
import numpy as np


# get filenames in songs directory
def abc_filenames(datapath='abcnotation_midi-test', count=False, example=False):
    files = [os.path.join(datapath, f) for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]
    # count songs
    if count==True:
        print('Found {} songs in directory'.format(len(files)))
    # example of a song in ABC notation
    if example==True:
        with open(files[1], 'r') as f:
            example_song = f.read()
        print('\nExample Song: ')
        print(example_song[1:-2])
    return files

# clean .abc files in songs directory
def clean_abc(filenames, count=False):
    i = 0
    songs = []

    for filename in filenames:
        f = open(filename,'r')
        #f = open(filename,'r',encoding='utf8')
        #f = open(filename,'r',encoding='ISO-8859-1')
        text = f.read() # original .abc

        # remove if it is an invalid song
        new_text = re.search('M:none',text)
        if hasattr(new_text, 'group'):
            continue

        # adjust order
        new_text = re.sub('(X:).*\n','X:'+str(i)+'\n',text)
        i += 1

        # remove title
        new_text = re.sub('(T:).*\n','',new_text)
        
        # remove lyrics
        new_text = re.sub('([wW]:).*\n','',new_text)
        
        # remove comments
        new_text = re.sub('(%).*\n','',new_text)

        # print results
        #print('original:\n',text)
        #print('\n after modifications:\n',new_text)

        # save songs after cleaning
        songs.append(new_text)

    # print how many songs are left
    if count==True:
        print('There will be used {} songs to train the model'.format(len(songs)))

    return songs

# join all songs in a single string
def join_songs(songs):
    songs_joined = []
    for song in songs:
        songs_joined.append(song[1:-2])
    songs_joined = '\n\n'.join(songs_joined)
    return songs_joined

# map char2idx and idx2char
def abc_mapIndex(songs, show=False):
    vocab = sorted(set(songs))
    char2idx = {u:i for i,u in enumerate(vocab)}
    idx2char = np.array(vocab)
    if show==True:
        print("\nThere are", len(vocab), "unique characters in the dataset")
        print('\n{')
        for char,_ in zip(char2idx, range(20)):
            print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
        print('  ...\n}')
    return [char2idx,idx2char]

# vectorize songs with mapIndex (string --> index)
def vectorize_string(string,char2idx,show=False):
    vectorized_list = np.array([char2idx[s] for s in string]) 
    if show==True:
        print ('{} ---- characters mapped to int ----> {}'.format(repr(string[:10]), vectorized_list[:10]))
        assert isinstance(vectorized_list, np.ndarray), "returned result should be a numpy array"
    return vectorized_list
