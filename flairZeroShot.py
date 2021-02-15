from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence
import spacy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

# 1. Load our pre-trained TARS model for English
tars = TARSClassifier.load('tars-base')

# 3. Define some classes that you want to predict using descriptive names
classes = ["positive sentiment", "negative sentiment"]

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
    sentenceObjList = [Sentence(x) for x in sentences if len(x)>0]
    
    #4. Predict for these classes
    tars.predict_zero_shot(sentenceObjList, classes)
    
    pos = 0
    neg = 0
    for el in sentenceObjList:
        if len(el.labels)>0:
            if "positive" in el.labels[0].value:
                pos += el.labels[0].score
            elif "negative" in el.labels[0].value:
                neg += el.labels[0].score
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
    outputList.append("pred: "+pred_label+"\ttrue: "+label[:3]+"\t"+text+"\n")

res = accuracy_score(true_labels, pred_labels)
print(res)
conf_matrix = confusion_matrix(true_labels, pred_labels)
print(conf_matrix)

with open("results/flair_zero_shot_results.txt", "w", encoding="utf-8") as r:
    r.write("Removed datapoints: "+str(removed_datapoints)+" of "+str(len(dataset))+"\n")
    r.write("Accuracy: "+str(res)+"\n")
    r.write(str(conf_matrix)+"\n")
    r.writelines(outputList)
