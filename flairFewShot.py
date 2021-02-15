from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence
import spacy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import argparse
from random import shuffle

n_classes = 2
shots = [1,2,5,10]

"""
---------- TRAINING ----------
"""
def train(k_shot):
    print("\nTraining "+str(k_shot)+"-shot")
    with open("datasets/imdb_train.txt", "r", encoding="utf-8") as d:
        dataset = [line[:-1] for line in d]
    
    shuffle(dataset)
    
    # select k examples for each of the n classes from the train set to train on
    train_samples = []
    t_pos_samples = 0
    t_neg_samples = 0
    dev_samples = []
    d_pos_samples = 0
    d_neg_samples = 0

    for el in dataset:
        if len(train_samples)==k_shot*n_classes and len(dev_samples)==n_classes:
            break
        label, text = el.split("\t", 1)
        if "positive" in label and t_pos_samples<k_shot:
            train_samples.append(Sentence(text).add_label("positive_or_negative", label))
            t_pos_samples += 1
        elif "negative" in label and t_neg_samples<k_shot:
            train_samples.append(Sentence(text).add_label("positive_or_negative", label))
            t_neg_samples += 1
        elif "positive" in label and d_pos_samples<(n_classes/2):
            dev_samples.append(Sentence(text).add_label("positive_or_negative", label))
            d_pos_samples += 1
        elif "negative" in label and d_neg_samples<(n_classes/2):
            dev_samples.append(Sentence(text).add_label("positive_or_negative", label))
            d_neg_samples += 1
        else:
            pass
    
    # training dataset consisting of k*n (n=2) sentences (k labeled as "positive" and k labeled as "negative")
    train = SentenceDataset(train_samples)
    
    # dev set of size n
    dev = SentenceDataset(dev_samples)
    
    # make a corpus with train and test split
    corpus = Corpus(train=train, dev=dev)
    
    # 1. load base TARS
    tars = TARSClassifier.load('tars-base')

    # 2. make the model aware of the desired set of labels from the new corpus
    tars.add_and_switch_to_new_task("POSITIVE_NEGATIVE", label_dictionary=corpus.make_label_dictionary())

    # 3. initialize the text classifier trainer with your corpus
    trainer = ModelTrainer(tars, corpus)

    # 4. train model
    trainer.train(base_path="models/flair/"+str(k_shot)+"_shot", # path to store the model artifacts
                  learning_rate=0.02, # use very small learning rate
                  mini_batch_size=1, # small mini-batch size since corpus is tiny
                  max_epochs=10, # terminate after 10 epochs
                  embeddings_storage_mode="gpu"
                  )

"""
---------- TESTING ----------
"""
def test(k_shot):
    print("\nTesting "+str(k_shot)+"-shot")
    # 1. Load few-shot TARS model
    tars = TARSClassifier.load("models/flair/"+str(k_shot)+"_shot/final-model.pt")

    with open("datasets/imdb_test.txt", "r", encoding="utf-8") as d:
        dataset = [line[:-1] for line in d]

    true_labels = []
    pred_labels = []
    outputList = []

    for datapoint in tqdm(dataset):
        
        label, text = datapoint.split("\t", 1)
        
        # 2. Prepare a test sentence
        sentence = Sentence(text)
        
        # 3. Predict for positive and negative
        tars.predict(sentence)
        
        pred_label = sentence.labels[0].value
        pred_labels.append(pred_label)
        true_labels.append(label)
        outputList.append("pred: "+pred_label+"\ttrue: "+label[:3]+"\t"+text+"\n")

    res = accuracy_score(true_labels, pred_labels)
    print(res)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(conf_matrix)

    with open("results/flair_"+str(k_shot)+"_shot_results.txt", "w", encoding="utf-8") as r:
        r.write("Accuracy: "+str(res)+"\n")
        r.write(str(conf_matrix)+"\n")
        r.writelines(outputList)


parser = argparse.ArgumentParser(description="Script to perform training or testing of k-shot text classification")
parser.add_argument("--train", choices=shots, type=int, help="Select k for k-shot training")
parser.add_argument("--test", choices=shots, type=int, help="Select k for k-shot testing. Only works if such a model is already trained")
args = parser.parse_args()

if args.train:
    train(vars(args)["train"])
if args.test:
    test(vars(args)["test"])

if __name__ == "__main__":
    for i in shots:
        train(i)
    for i in shots[2:]:
        test(i)
    