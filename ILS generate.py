import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from stackRNN import StackAugmentedRNN
from data import GeneratorData
from util import canonical_smiles
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sys.path.append('./release/')

use_cuda = torch.cuda.is_available()
gen_data_path = './smiles_all.smi'
tokens = ['<', '>' ,'#',  ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7','T',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'L','M', 'N', 'P', 'S', 'G', '[', ']',
          '\\', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'o', 'n', 'p', 's', 't','u', 'r', 'Z', '-', '.']
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)
def plot_hist(prediction, n_to_generate):
    print("Mean value of predictions:", prediction.mean())
    print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
    ax = sns.kdeplot(prediction, shade=True)
    ax.set(xlabel='Predicted pIC50', 
           title='Distribution of predicted pIC50 for generated molecules')
    plt.show()

hidden_size = 500
stack_width = 500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta


my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, layer_type=layer_type,
                                 n_layers=1, is_bidirectional=False, has_stack=True,
                                 stack_width=stack_width, stack_depth=stack_depth, 
                                 use_cuda=use_cuda, 
                                 optimizer_instance=optimizer_instance, lr=lr)

losses = my_generator.fit(gen_data, 1000000)

plt.plot(losses)
plt.savefig('Training loss.png', dpi=300)
model_path = '/root/Code/分子生成/ILs生成/final_smile_model/checkpoint_biggest_rnn'
my_generator.evaluate(gen_data)

my_generator.load_model(model_path)

def estimate_and_update(generator, n_to_generate, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=200)[1:-1])

    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
#     smiles, prediction, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)  
                                                       
#     plot_hist(prediction, n_to_generate)
        
    return unique_smiles

smiles_cur= estimate_and_update(my_generator, n_to_generate=1000)
pd.DataFrame(smiles_cur).to_excel('smiple_smiles_nostack.xlsx')
print('Sample trajectories:')
for sm in smiles_cur[:]:
    print(sm)