import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import gzip
import csv
from torch.nn.utils.rnn import pack_padded_sequence
from rdkit import Chem
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import train_test_split

hidden_size=200
n_layers=2
n_epoch=500
batch_size=16
USE_GPU = False
set_cv = 5
def createTensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device) 
    return tensor

class RNNRegress(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, bidirectional = True):
        super(RNNRegress, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # embedding
        # input size: (seq_len, batch_size)
        # output size: (seq_len, batch_size, hidden_size)
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # GRU
        # INPUTS: input size: (seq_len, batch_size, hidden_size)
        # INPUTS: hidden size: (num_layers * num_directions, batch_size, hidden_size)
        # OUTPUTS: output size: (seq_len, batch_size, hidden_size * num_directions)
        # OUTPUTS: hidden size: (num_layers * num_directions, batch_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers, bidirectional = bidirectional)

        self.fc = torch.nn.Linear(hidden_size * self.num_directions, output_size)

    def initHidden(self, batch_size):
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        return createTensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t() 
        batch_size = input.size(1)
        hidden = self.initHidden(batch_size) # init hidden 0

        # embedding layer
        embedding = self.embedding(input) # embedding size: batch_size, seq_len, embedding_size
        # GRU
        gru_input = pack_padded_sequence(embedding, seq_lengths) 
        output, hidden = self.gru(gru_input, hidden)
        if self.num_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim = 1)
        else:
            hidden_cat = hidden[-1] 

        fc_output = self.fc(hidden_cat)
        return fc_output
    # smiles to ASCIIlist
def smiles_to_ASCIIlist(smiles_list):
    ASCIIlist = [ord(smiles) for smiles in smiles_list]
    return ASCIIlist

# sort and padding 
def makeTensors(smiles_list, tox_list):
    smiles_sequences = [smiles_to_ASCIIlist(smiles) for smiles in smiles_list]
    smiles_seq_lens = torch.LongTensor([len(name_ASCII) for name_ASCII in smiles_sequences])

    # padding make tensor of smiles, BatchSize * SeqLen 
    smiles_tensor = torch.zeros(len(smiles_sequences), smiles_seq_lens.max()).long()
    for index, (smiles_sequence, smiles_seq_len) in enumerate(zip(smiles_sequences, smiles_seq_lens), 0):
        smiles_tensor[index, 0:smiles_seq_len] = torch.LongTensor(smiles_sequence)

    # sort by length for pack_padded_sequence
    ordered_smiles_seq_lens, len_indexes = smiles_seq_lens.sort(dim = 0, descending = True)
    ordered_smiles_tensor = smiles_tensor[len_indexes]
    ordered_tox_list = tox_list[len_indexes]

    return createTensor(ordered_smiles_tensor), createTensor(ordered_smiles_seq_lens), createTensor(ordered_tox_list)

def train():
    loss = 0.0
    for batch_index, (smiles, tox) in enumerate(train_loader): 
        inputs, seq_lens, targets = makeTensors(smiles, tox) 
        outputs = classifier_model(inputs, seq_lens)
        targets = targets.to(torch.float32).reshape(-1,1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss += loss.item()

        if batch_index % 3 == 2:
            print('time_elapsed: {:.1f}m {:.1f}s, Epoch {}, '.format(timePassed(start_time)[0], timePassed(start_time)[1], epoch), end = '')
            print(f'[{(batch_index+1) * len(inputs)} / {len(train_set_5cv)}] ', end = '')
            print(f'loss = {loss / ((batch_index+1) * len(inputs))}')

    return loss.detach().numpy()/len(train_loader)

def test():
    correct = 0
    with torch.no_grad():
        pred = []
        obs = []
        for i, (smiles, tox) in enumerate(test_loader):
            inputs, seq_lens, targets = makeTensors(smiles, tox)
            outputs = classifier_model(inputs, seq_lens)
            pred.append(outputs.data)
            obs.append(targets.data.reshape(-1,1))
        pred = np.vstack(pred).ravel()
        obs = np.vstack(obs).ravel()
        r2 = r2_score(obs, pred)
        Mse = mse(obs, pred)
        Mae = mae(obs,pred)
        print('r2 on test: %.3f %%' % (100 * r2), 'mse on test: %.5f \n' % (Mse),'mae on test: %.5f \n' % (Mae))
    return r2, Mse,Mae, pred, obs

def timePassed(start_time):
    time_passed = time.time() - start_time
    minute = math.floor(time_passed / 60)
    second = time_passed - minute * 60
    return [minute, second]


# data load
tox = pd.read_excel(r'./data.xlsx')
data_tox =tox.iloc[:,-1]
data_mols = Chem.SDMolSupplier("./ACHE.sdf")
data_smiles = [Chem.MolToSmiles(mol,kekuleSmiles=True) for mol in data_mols]
data_smiles_len = [len(smiles) for smiles in data_smiles]
data_all = [(data_smiles[i], data_tox[i]) for i in range(len(data_smiles))]

data_tox 

#Data split
# train , test = train_test_split(data_all,test_size = 0.2)
kfold_train = KFold(n_splits = 5 ,shuffle = True)
r2_best = []
for fold, (train_idx, val_idx) in enumerate(kfold_train.split(data_all)):
    print('**'*10,'第', fold+1, '折','ing....', '**'*10)
    train_set_5cv = [data_all[index] for index in train_idx]
    test_set_5cv = [data_all[index] for index in val_idx]
    
    # parameters
    n_chars = 0
    for smiles in data_smiles:
        for char in smiles:
            if n_chars < ord(char):
                n_chars = ord(char)
    n_chars = n_chars+1
    output_size = 1


    train_loader = DataLoader(dataset = train_set_5cv, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_set_5cv, batch_size = batch_size, shuffle = False)



    classifier_model = RNNRegress(n_chars, hidden_size, output_size, n_layers) # input_size, hidden_size, output_size, num_layers
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier_model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr = 0.001, weight_decay=10 ** (-5.0))

    # training and test
    start_time = time.time()
    print("The num of total training epochs is %d. " % n_epoch)
    r2_list = []
    Mse_list = []
    Mae_list = []
    best_r2 = 0
    train_loss = []
    
    for epoch in range(n_epoch):
        train_loss.append(train())
        r2, Mse,Mae, pred, obs = test()
        r2_list.append(r2)
        Mse_list.append(Mse)
        Mae_list.append(Mae)
        if best_r2 < r2:
            best_r2 = r2
            result ={"obs":obs,"pred":pred}
            pd.DataFrame(result).to_excel('./5cv_results/result{}.xlsx'.format(fold+1))
            
    plt.plot(np.arange(1, n_epoch+1), r2_list)
    plt.plot(np.arange(1, n_epoch+1), Mae_list)
    plt.plot(np.arange(1, n_epoch+1), Mse_list)
    plt.xlabel("epoch")
    plt.ylabel("r2 & MSE & MAE")
    plt.grid()
    plt.savefig('./5cv_results/result_{}cv.png'.format(fold+1),dpi=300)
    plt.show()

    pd.DataFrame(r2_list).to_excel('./5cv_results/r2_list_{}cv.xlsx'.format(fold+1))
    pd.DataFrame(Mse_list).to_excel('./5cv_results/mse_list_{}cv.xlsx'.format(fold+1))
    pd.DataFrame(train_loss).to_excel('./5cv_results/train_loss_{}cv.xlsx'.format(fold+1))
    pd.DataFrame(Mae_list).to_excel('./5cv_results/MAE_list_{}cv.xlsx'.format(fold+1))
    print('r2_max:',max(r2_list))
    best_r2 = max(r2_list)
    torch.save(classifier_model.state_dict(),'./5cv_results/GRU_SMILES_model{}.pt'.format(fold+1))
    r2_best.append(best_r2 )
    r2_list.clear()
    Mse_list.clear()
    Mae_list.clear()
    train_loss.clear()
print('r2_max_mean{}_cv:'.format(fold+1), sum(r2_best)/len(r2_best))
pd.DataFrame(r2_best).to_excel('./5cv_results/result_r2_best_list.xlsx')
    