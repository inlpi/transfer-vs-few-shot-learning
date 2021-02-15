from transformers import pipeline
import spacy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

# 1. Load our pre-trained TARS model for English
classifier = pipeline("zero-shot-classification", device=0)

# 3. Define some classes that you want to predict using descriptive names
classes = ["positive", "negative"]

# load data

with open("datasets/imdb_test.txt", "r", encoding="utf-8") as d:
    dataset = [line[:-1] for line in d]

true_labels = []
pred_labels = []
outputList = []
removed_datapoints = 0

for datapoint in tqdm(dataset):
    
    label, text = datapoint.split("\t", 1)
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # 2. Prepare a test sentence
    classifiedObjList = [classifier(x, classes) for x in sentences if len(x)>0]
    
    pos = 0
    neg = 0
    for el in classifiedObjList:
        if "positive" in el["labels"][0]:
            pos += el["scores"][0]
        elif "negative" in el["labels"][0]:
            neg += el["scores"][0]
        else:
            print("Error")
    
    pred_label = ""
    
    if pos > neg:
        pred_labels.append("positive sentiment")
        pred_label = "pos"
    elif neg > pos:
        pred_labels.append("negative sentiment")
        pred_label = "neg"
    else:
        removed_datapoints += 1
        continue
    
    true_labels.append(label)
    outputList.append("pred: "+pred_label[:3]+"\ttrue: "+label[:3]+"\t"+text+"\n")

res = accuracy_score(true_labels, pred_labels)
print(res)
conf_matrix = confusion_matrix(true_labels, pred_labels)
print(conf_matrix)

with open("results/hug_sentencewise_zero_shot_results.txt", "w", encoding="utf-8") as r:
    r.write("Removed datapoints: "+str(removed_datapoints)+" of "+str(len(dataset))+"\n")
    r.write("Accuracy: "+str(res)+"\n")
    r.write(str(conf_matrix)+"\n")
    r.writelines(outputList)
