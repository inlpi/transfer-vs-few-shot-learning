import ast
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

device = "cuda:0" if torch.cuda.is_available() else "cpu"

BASE_MODEL = 'distilbert-base-cased'

BATCH_SIZE = 5
WARMUP_EPOCHS = 0
TRAIN_EPOCHS = 10


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, test=True):
        
        dataset = []
        
        # if test=True, we load the test set for testing
        if test:
            with open("datasets/imdb_test.txt", "r", encoding="utf-8") as d:
                dataset = [line[:-1] for line in d]
        # if test=False, we load the train set for training
        else:
            with open("datasets/imdb_train.txt", "r", encoding="utf-8") as d:
                dataset = [line[:-1] for line in d]
        
        self.data = []
        self.labels = []
        
        for datapoint in dataset:
            label, text = datapoint.split("\t", 1)
            self.data.append(text)
            if "positive" in label:
                self.labels.append(1)
            elif "negative" in label:
                self.labels.append(0)
            else:
                raise ValueError("wrong label")
        
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X,y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)


def train(freeze, batch_size=BATCH_SIZE, warmup_epochs=WARMUP_EPOCHS, train_epochs=TRAIN_EPOCHS):
   
    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels = 2, output_attentions = False, output_hidden_states = False)
    
    if freeze:
        for param in model.base_model.parameters():
            param.requires_grad = False
    model = model.to(device)
    model.train()
    train_set = SentimentDataset(test=False)
    print("Training set length:", len(train_set))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    n_batches = len(trainloader)
    warmup_steps = warmup_epochs * n_batches
    train_steps = n_batches * train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    
    for _ in tqdm(range(-warmup_epochs, train_epochs)):
        for sents, labels in tqdm(trainloader):
            optimizer.zero_grad()
            labels = labels.to(device)
            
            input_ids = tokenizer(sents, padding=True, truncation=True, is_split_into_words=False, return_tensors='pt', return_attention_mask=False).to(device)
            loss, predictions = model(**input_ids, labels=labels)
            loss.backward()
   
            optimizer.step()
            scheduler.step()
    
    if freeze:
        model.save_pretrained("models/hug/"+BASE_MODEL+"_seq_frozen_encoder")
    else:
        model.save_pretrained("models/hug/"+BASE_MODEL+"_seq_fully_finetuned")

def test(model=None, model_path=None):
    out_file = ""
    outputList = []
    
    if BASE_MODEL+"_seq_frozen_encoder" in model_path:
        out_file = BASE_MODEL+"_seq_frozen_encoder"
    elif BASE_MODEL+"_seq_fully_finetuned" in model_path:
        out_file = BASE_MODEL+"_seq_fully_finetuned"
    else:
        out_file = BASE_MODEL+"_seq_zero_shot"
    
    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)
    if model is None:
        model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels = 2, output_attentions = False, output_hidden_states = False)
    model = model.to(device)
    model.eval()
    test_set = SentimentDataset(test=True)
    print("Test set length:", len(test_set))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=False)
   
    pred_list, label_list = list(), list()
    for sents, labels in tqdm(testloader):
        labels = labels.to(device)
        
        input_ids = tokenizer(sents, padding=True, truncation=True, is_split_into_words=False, return_tensors='pt', return_attention_mask=False).to(device)
        loss, predictions = model(**input_ids, labels=labels)
        
        preds = list(np.argmax(predictions.detach().tolist(), axis=1))
        labls = labels.detach().tolist()
        pred_list.extend(preds)
        label_list.extend(labls)
        
        for p,l,s in zip(preds, labls, sents):
            pred_label = ""
            if p==0:
                pred_label = "neg"
            elif p==1:
                pred_label = "pos"
            else:
                print("Error")
            true_label = ""
            if l==0:
                true_label = "neg"
            elif l==1:
                true_label = "pos"
            else:
                print("Error")
            outputList.append("pred: "+pred_label+"\ttrue: "+true_label+"\t"+s+"\n")
        
    conf_matrix = confusion_matrix(label_list, pred_list)
    class_report = classification_report(label_list, pred_list, target_names=["neg", "pos"])
    print(conf_matrix)
    print(class_report)
    
    with open("results/hug_"+out_file+"_results.txt", "w", encoding="utf-8") as r:
        r.write(str(conf_matrix)+"\n")
        r.write(str(class_report)+"\n")
        r.writelines(outputList)
    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    parser = argparse.ArgumentParser(description="Script to fine-tune a BERT model for multiple choice")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", choices=["frozen-encoder", "fully-finetuned"],
                      help="Sets the mode how to fine-tune the model")
    mode.add_argument("--test", choices=["zero-shot", "frozen-encoder", "fully-finetuned"],
                      help="Sets the mode how to test the model")
    args = parser.parse_args()

    if args.train:
        if args.train == "frozen-encoder":
            train(freeze=True)
        elif args.train == "fully-finetuned":
            train(freeze=False)
        else:
            raise Exception("Unknown or empty argument:", args.train)
    else:
        if args.test == "zero-shot":
            test(model_path=BASE_MODEL)
        elif args.test == "frozen-encoder":
            test(model_path="models/hug/"+BASE_MODEL+"_seq_frozen_encoder")
        elif args.test == "fully-finetuned":
            test(model_path="models/hug/"+BASE_MODEL+"_seq_fully_finetuned")
        else:
            raise Exception("Unknown or empty argument:", args.test)
