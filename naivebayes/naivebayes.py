import numpy as np


def setup():
    pass

def p1():
    for lang in ['e', 's', 'j']:
        for i in range(0,20):
            filename = './languageID/'+lang+str(i)+'.txt'
            f = open(filename, 'r')
            print(filename)
            print(f.readlines())

#setup()
p1()
