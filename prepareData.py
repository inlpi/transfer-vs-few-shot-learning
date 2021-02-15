import os
from random import shuffle

for dset in ["train", "test"]:
    pos = []
    neg = []
    
    for filename in os.listdir("data/aclImdb_v1/"+dset+"/pos/"):
        with open("data/aclImdb_v1/"+dset+"/pos/"+filename, "r", encoding="utf-8") as f:
            text = f.read().replace("<br />", "")
            pos.append("positive sentiment\t"+text+"\n")

    for filename in os.listdir("data/aclImdb_v1/"+dset+"/neg/"):
        with open("data/aclImdb_v1/"+dset+"/neg/"+filename, "r", encoding="utf-8") as f:
            text = f.read().replace("<br />", "")
            neg.append("negative sentiment\t"+text+"\n")

    data = pos + neg
    shuffle(data)
    
    with open("imdb_"+dset+".txt", "w", encoding="utf-8") as r:
        r.writelines(data)
