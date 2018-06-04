import nltk

c = nltk.corpus.cmudict.dict()
keys = c.keys()

for key in keys:
    for i in range(0,len(c[key])):
        previous = '-1'
        for j in range(0,len(c[key][i])):
            if previous == c[key][i][j]:
                print(key)
            previous = c[key][i][j]
