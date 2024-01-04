


import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.Draw import DrawMorganBit, DrawMorganBits, DrawMorganEnv, IPythonConsole, rdMolDraw2D

def mols_to_FP(mols, radius=3, nBits=1024, useFeatures=False):
    l = len(mols)
    mfp_pd = np.zeros((l, nBits), dtype='int')
    bitInfo_all = []
    for i in range(l):
        bitInfo={}
        mfp = AllChem.GetMorganFingerprintAsBitVect(mols[i], radius, nBits, useFeatures=useFeatures, bitInfo=bitInfo)
        mfp_pd[i,:]=np.array(list(mfp.ToBitString()))
        bitInfo_all.append(bitInfo)
        del bitInfo
    return mfp_pd, bitInfo_all
def MACCS (mol):
    molToMA = [list(AllChem.GetMACCSKeysFingerprint(mol)) for mol in mols_train]
    return molToMA
#Load Data
mols_train = Chem.SDMolSupplier(r'./AChE_CATION.sdf')
ECFP, ECFP_bitInfo_all = mols_to_FP(mols_train, useFeatures=False)
FCFP, FCFP_bitInfo_all = mols_to_FP(mols_train, useFeatures=True)
TO_MACCS =MACCS(mols_train) 

# Save the results
name = pd.DataFrame([mol.GetProp('_Name') for mol in mols_train],columns=['smiles'])
pd.concat([name,pd.DataFrame(ECFP)],axis=1).to_excel('ECFP_cation.xlsx')
# pd.concat([name,pd.DataFrame(FCFP)],axis=1).to_excel('FCFP.xlsx')
# pd.concat([name,pd.DataFrame(TO_MACCS)],axis=1).to_excel('MCFP.xlsx')