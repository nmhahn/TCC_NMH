import os
import re
from functions.reader import *

filenames = abc_filenames()[0:10]

def clean_abc(filenames):
    n = len(filenames)
    i = 0

    for filename in filenames:
        f = open(filename,'r',encoding='utf8')
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
        print('original:\n',text)
        print('\n after modifications:\n',new_text)


if __name__ == '__main__':
    clean_abc(filenames)