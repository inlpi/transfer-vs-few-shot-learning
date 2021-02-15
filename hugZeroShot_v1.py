from transformers import pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

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

for datapoint in tqdm(dataset):
    
    label, text = datapoint.split("\t", 1)
    
    #4. Predict for these classes
    classified = classifier(text, classes)
    
    pred_label = classified["labels"][0]
    
    pred_labels.append(pred_label)
    true_labels.append(label.split(" ")[0])
    outputList.append("pred: "+pred_label[:3]+"\ttrue: "+label[:3]+"\t"+text+"\n")

res = accuracy_score(true_labels, pred_labels)
print(res)
conf_matrix = confusion_matrix(true_labels, pred_labels)
print(conf_matrix)

with open("results/hug_v1_zero_shot_results.txt", "w", encoding="utf-8") as r:
    r.write("Accuracy: "+str(res)+"\n")
    r.write(str(conf_matrix)+"\n")
    r.writelines(outputList)
