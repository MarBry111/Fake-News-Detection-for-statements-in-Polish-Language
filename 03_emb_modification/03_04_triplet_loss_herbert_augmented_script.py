import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.base import clone as sklearn_clone

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score,
    precision_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, HerbertTokenizer, BatchEncoding

import gc
from sklearn.neighbors import KNeighborsClassifier

import scipy
from scipy import spatial


torch.manual_seed(111)
np.random.seed(111)
random.seed(111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()
    
print(device.type)


df_topics = pd.read_csv('../datasets/ready2use/topics.csv', index_col=0)

df = pd.read_csv('../datasets/ready2use/fake_news_features_combined.csv', sep=';')

df = df[ df['assestment'] != 'brak' ]

df.loc[:, 'assestment'] = df['assestment'].replace({
    'falsz' : 'Fałsz',
    'zbity_zegar' : 'Fałsz',
    'raczej_falsz' : 'Fałsz',
    'prawda' : 'Prawda',
    'blisko_prawdy' : 'Prawda',
    'polprawda' : 'Manipulacja',
    'Częściowy fałsz' : 'Manipulacja'
})

df = df[ df['assestment'] != 'Nieweryfikowalne' ]
df = df[ df['assestment'] != 'Manipulacja' ]

df['assestment'] = df['assestment'].replace({
    'Fałsz' : 0,
#     'Manipulacja' : 1,
    'Prawda' : 1
}).astype(int)

df = df.copy()[['assestment', 'text_clean']][df.index.isin(df_topics.index)].reset_index(drop=True)

with open('../datasets/ready2use/style_emb_pl.npy', 'rb') as f:
    emb_style = np.load(f)
    
embeddings_table = pd.read_csv('../datasets/ready2use/embeddings_pl_herbert.csv', sep=",", header=None).values

with open('../datasets/ready2use/embeddings_pl_herbert_aug.npy', 'rb') as f:
    embeddings_table_aug = np.load(f)
    

cv_fold = []
cv_fold_i = []

for i in df_topics['topic'].unique().reshape(10,-1):
    train_cv = df_topics.index[ ~np.isin(df_topics["topic"], [i, np.mod(i+1,10)]) ].values
    val_cv = df_topics.index[ ~np.isin(df_topics["topic"], np.mod(i+1,10)) ].values
    test_cv = df_topics.index[ np.isin(df_topics["topic"], i) ].values
    
    train_cv_i = df_topics.reset_index().index[ ~np.isin(df_topics["topic"], [i, np.mod(i+1,10)]) ].values
    val_cv_i = df_topics.reset_index().index[ ~np.isin(df_topics["topic"], np.mod(i+1,10)) ].values
    test_cv_i = df_topics.reset_index().index[ np.isin(df_topics["topic"], i) ].values
    
    cv_fold.append( [train_cv, val_cv, test_cv])
    cv_fold_i.append( [train_cv_i, val_cv_i, test_cv_i])
    
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(df_topics)

cv_Kfold = []
cv_Kfold_i = []

for train_index, test_index in kf.split(df_topics):
    train_index, val_index = train_test_split(train_index, test_size=1/9, shuffle=True)
    train_cv = df_topics.iloc[ train_index, : ].index.values
    val_cv = df_topics.iloc[ val_index, : ].index.values
    test_cv = df_topics.iloc[ test_index, : ].index.values

    train_cv_i= df_topics.reset_index().iloc[ train_index, : ].index.values
    val_cv_i = df_topics.reset_index().iloc[ val_index, : ].index.values
    test_cv_i = df_topics.reset_index().iloc[ test_index, : ].index.values
    
    cv_Kfold.append( [train_cv, val_cv, test_cv])
    cv_Kfold_i.append( [train_cv_i, val_cv_i, test_cv_i])
    
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
    
    
class FakeNews(Dataset):
    def __init__(self, emb_dt, y_dt, index_to_use, factor=1):
        self.emb = emb_dt[index_to_use]
        
        self.labels = y_dt[index_to_use]
        
        self.index = np.repeat( [np.arange(index_to_use.shape[0])], factor, axis=0).reshape(-1)
        
        self.idx = np.arange(index_to_use.shape[0])
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx): 
        item = self.index[idx]
                
        anchor_label = self.labels[item]

        positive_list = self.idx[self.idx!=item][self.labels[self.idx!=item]==anchor_label]

        positive_item = random.choice(positive_list)
        
        negative_list = self.idx[self.idx!=item][self.labels[self.idx!=item]!=anchor_label]
        negative_item = random.choice(negative_list)
        
        anchor_claim = self.emb[item].astype(np.float32)
        positive_claim = self.emb[positive_item].astype(np.float32)
        negative_claim = self.emb[negative_item].astype(np.float32)

        anchor_label = anchor_label.astype(np.float32)

        return anchor_claim, positive_claim, negative_claim, anchor_label

class FakeNewsLabel(Dataset):
    def __init__(self, emb_dt, y_dt, index_to_use):
        self.emb = emb_dt[index_to_use]
        
        self.labels = y_dt[index_to_use]
        
        self.index = np.arange(index_to_use.shape[0])
        
    def __len__(self):
        return len(self.emb)
    
    def __getitem__(self, item):
        anchor_label = self.labels[item]
        
        anchor_claim = self.emb[item].astype(np.float32)

        anchor_label = anchor_label.astype(np.float32)

        return anchor_claim, anchor_label
    
    

embedding_dims = 20
batch_size = 512
epochs = 200
n_workers = 8



def prepare_loaders(emb_tab=embeddings_table, 
                    emb_tab_train=embeddings_table_aug,
                    y_column=df['assestment'].values, 
                    batch_size=batch_size,
                    index_llist = [train_index, val_index, test_index],
                    sizes= [10,3],
                    n_workers=n_workers
                   ):
    train_index, val_index, test_index = index_llist
    
    train_index_more = np.concatenate([train_index.reshape(-1, 1)*10+i for i in range(10)], 1).reshape(-1)
    
    train_ds = FakeNews(emb_tab_train, np.repeat(y_column, 10), train_index_more, 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    train_ds_label = FakeNewsLabel(emb_tab, y_column, train_index)
    train_loader_label = DataLoader(train_ds_label, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    val_ds = FakeNews(emb_tab, y_column, val_index, 1)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    test_ds = FakeNewsLabel(emb_tab, y_column, test_index)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    return train_loader, train_loader_label, val_loader, test_loader



class Network(nn.Module):
    def __init__(self, emb_dim=128, in_shape=1024):
        super(Network, self).__init__()
        
        self.in_shape = in_shape
        
        self.fc = nn.Sequential(
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
            
            nn.Linear(in_shape, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, emb_dim)
            
#             nn.Linear(1024, emb_dim),
#             nn.BatchNorm1d(emb_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
        )
        
    def forward(self, x):
        x = x.view(-1, self.in_shape)
        x = self.fc(x)
        return x
    
    
    
def train_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1000,
    n_print=100,
    if_norm=False,
    optimizer=optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
    criterion=TripletLoss(),
    n_model=0,
    model_name='model_X.pt'
):
    val_prev = np.inf
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        for step, (anchor_claim, positive_claim, negative_claim, anchor_label) in enumerate(train_loader):
            anchor_claim = anchor_claim.to(device) if if_norm else (anchor_claim / torch.norm(anchor_claim) ).to(device)
            positive_claim = positive_claim.to(device) if if_norm else (positive_claim / torch.norm(positive_claim) ).to(device)
            negative_claim = negative_claim.to(device) if if_norm else (negative_claim / torch.norm(negative_claim) ).to(device)
            
            optimizer.zero_grad()
            anchor_out = model(anchor_claim)
            positive_out = model(positive_claim)
            negative_out = model(negative_claim)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        model.eval()
        
        val_loss = []
        for anchor_claim, positive_claim, negative_claim, _ in val_loader:
            anchor_claim = anchor_claim.to(device) if if_norm else (anchor_claim / torch.norm(anchor_claim) ).to(device)
            positive_claim = positive_claim.to(device) if if_norm else (positive_claim / torch.norm(positive_claim) ).to(device)
            negative_claim = negative_claim.to(device) if if_norm else (negative_claim / torch.norm(negative_claim) ).to(device)

            anchor_out = model(anchor_claim)
            positive_out = model(positive_claim)
            negative_out = model(negative_claim)

            loss = criterion(anchor_out, positive_out, negative_out)
            val_loss.append(loss.cpu().detach().numpy())

        model.train()
        
        if np.mean(val_loss) < val_prev:
            val_prev = np.mean(val_loss)
            torch.save(model, f'models/{model_name}')

        if epoch%n_print == 0:
            print(f"{n_model} Epoch: {epoch+1}/{epochs} - Train Loss: {np.mean(running_loss):.4f};",
                  f" Val Loss: {np.mean(val_loss):.4f} Best Val loss {val_prev:.4f}")

            
            
            
results = {
    'test_accuracy' : [],
    'test_precision' : [],
    'test_recall' : [],
    'test_f1' : []
}

embedding_dims = 100
batch_size = 512
epochs = 1000
n_workers = 8

for j, (train_index, val_index, test_index) in enumerate(cv_fold_i):
    train_loader, train_loader_label, val_loader, test_loader = prepare_loaders(
        emb_tab=embeddings_table, 
        emb_tab_train=embeddings_table_aug,
        y_column=df['assestment'].values, 
        batch_size=batch_size,
        index_llist = [train_index, val_index, test_index],
        sizes= [10,3],
        n_workers=n_workers,
        n_model=j
    )
    
    model = Network(embedding_dims)
    model = model.to(device)
    
    if_norm = False 
    
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1000,
        n_print=100,
        if_norm=if_norm,
        optimizer=optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
        criterion=TripletLoss(),
        n_model=0,
        model_name='model_X.pt'
    )
    
    model = torch.load('models/model_X.pt')
    
    
    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for anchor_claim, label in tqdm(train_loader_label):
            anchor_claim = anchor_claim.to(device) if if_norm else (anchor_claim / torch.norm(anchor_claim) ).to(device)

            train_results.append(model(anchor_claim).cpu().numpy())
            labels.append(label)

    train_results = np.concatenate(train_results) 
    labels = np.concatenate(labels)
    train_results.shape

    
    test_results = []
    test_labels = []

    model.eval()
    with torch.no_grad():
        for anchor_claim, label in tqdm(test_loader):
            anchor_claim = anchor_claim.to(device) if if_norm else (anchor_claim / torch.norm(anchor_claim) ).to(device)

            test_results.append(model(anchor_claim).cpu().numpy())
            test_labels.append(label)

    test_results = np.concatenate(test_results)
    test_labels = np.concatenate(test_labels)
    test_results.shape

    
    clf_lr_1 = LogisticRegression(max_iter=5000, C=1, penalty='l2', solver='liblinear')

    y_train_t = labels
    X_train_t = train_results
    y_test_t = test_labels
    X_test_t = test_results

    clf_lr_1.fit(X_train_t, y_train_t)

    y_pred = clf_lr_1.predict(X_test_t)

    results['test_accuracy'].append( accuracy_score(y_test_t, y_pred) ) 
    results['test_precision'].append( precision_score(y_test_t, y_pred) ) 
    results['test_recall'].append( recall_score(y_test_t, y_pred) ) 
    results['test_f1'].append( f1_score(y_test_t, y_pred) ) 


out = {
    "Accuracy": np.array(results['test_accuracy']),
#     "Precision": np.array(results['test_precision']).mean(),
#     "Recall": np.array(results['test_recall']).mean(),
    "F1 Score":  np.array(results['test_f1']),
    }

print()
print(
    'triplet loss lr C1',
    f'Accuracy {out["Accuracy"].mean():.3f}+-{out["Accuracy"].std():.3f}',
    f'F1 Score {out["F1 Score"].mean():.3f}+-{out["F1 Score"].std():.3f}',
    f' {out["Accuracy"].mean():.3f}+-{out["Accuracy"].std():.3f} | {out["F1 Score"].mean():.3f}+-{out["F1 Score"].std():.3f}'
)    



print()
print()
print()

results = {
    'test_accuracy' : [],
    'test_precision' : [],
    'test_recall' : [],
    'test_f1' : []
}

embedding_dims = 100
batch_size = 512
epochs = 1000
n_workers = 8

for j, (train_index, val_index, test_index) in enumerate(cv_Kfold_i):
    train_loader, train_loader_label, val_loader, test_loader = prepare_loaders(
        emb_tab=embeddings_table, 
        emb_tab_train=embeddings_table_aug,
        y_column=df['assestment'].values, 
        batch_size=batch_size,
        index_llist = [train_index, val_index, test_index],
        sizes= [10,3],
        n_workers=n_workers
    )
    
    model = Network(embedding_dims)
    model = model.to(device)

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1000,
        n_print=100,
        if_norm=False,
        optimizer=optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
        criterion=TripletLoss(),
        n_model=0,
        model_name='model_X.pt'
    )
    
    model = torch.load('models/model_X.pt')
    
    
    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for anchor_claim, label in tqdm(train_loader_label):
            anchor_claim = anchor_claim.to(device) if if_norm else (anchor_claim / torch.norm(anchor_claim) ).to(device)

            train_results.append(model(anchor_claim).cpu().numpy())
            labels.append(label)

    train_results = np.concatenate(train_results) 
    labels = np.concatenate(labels)
    train_results.shape

    
    test_results = []
    test_labels = []

    model.eval()
    with torch.no_grad():
        for anchor_claim, label in tqdm(test_loader):
            anchor_claim = anchor_claim.to(device) if if_norm else (anchor_claim / torch.norm(anchor_claim) ).to(device)

            test_results.append(model(anchor_claim).cpu().numpy())
            test_labels.append(label)

    test_results = np.concatenate(test_results)
    test_labels = np.concatenate(test_labels)
    test_results.shape

    
    clf_lr_1 = LogisticRegression(max_iter=5000, C=1, penalty='l2', solver='liblinear')

    y_train_t = labels
    X_train_t = train_results
    y_test_t = test_labels
    X_test_t = test_results

    clf_lr_1.fit(X_train_t, y_train_t)

    y_pred = clf_lr_1.predict(X_test_t)

    results['test_accuracy'].append( accuracy_score(y_test_t, y_pred) ) 
    results['test_precision'].append( precision_score(y_test_t, y_pred) ) 
    results['test_recall'].append( recall_score(y_test_t, y_pred) ) 
    results['test_f1'].append( f1_score(y_test_t, y_pred) ) 


out = {
    "Accuracy": np.array(results['test_accuracy']),
#     "Precision": np.array(results['test_precision']).mean(),
#     "Recall": np.array(results['test_recall']).mean(),
    "F1 Score":  np.array(results['test_f1']),
    }

print()
print(
    'triplet loss lr C1',
    f'Accuracy {out["Accuracy"].mean():.3f}+-{out["Accuracy"].std():.3f}',
    f'F1 Score {out["F1 Score"].mean():.3f}+-{out["F1 Score"].std():.3f}',
    f' {out["Accuracy"].mean():.3f}+-{out["Accuracy"].std():.3f} | {out["F1 Score"].mean():.3f}+-{out["F1 Score"].std():.3f}'
)    
