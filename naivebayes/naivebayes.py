import numpy as np


def setup():
    pass


def read_in_language_files(lang, min, max):
    words = []
    for i in range(min,max+1):
        filename = './languageID/'+lang+str(i)+'.txt'
        f = open(filename, 'r')
        lines = f.readlines()
        for line in lines:
            if not line.isspace():
                words.append(line.strip('\n'))
    return "".join([str(item) for item in words])

def char_to_idx(c):
    return ord(c) - 96

def get_lang_histogram(language_chars):
    abc = [0] * 27
    for c in language_chars:
        if c == ' ':
            abc[0] = abc[0] + 1
        else:
            abc[char_to_idx(c)] = abc[char_to_idx(c)] + 1
    return abc

def p1():
    e10 = read_in_language_files('e', 0, 9)
    s10 = read_in_language_files('s', 0, 9)
    j10 = read_in_language_files('j', 0, 9)

    print([chr(ord('a') + i) for i in range(0,26)])
    print(get_lang_histogram(e10))
    print(get_lang_histogram(s10))
    print(get_lang_histogram(j10))

#setup()
p1()
