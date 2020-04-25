#
# @author: Allan
#
from typing import List

from config.reader import  Reader
import numpy as np
import transformers
from transformers.modeling_distilbert import DistilBertModel
from tokenizers import BertWordPieceTokenizer

from torch.utils.data import Dataset, DataLoader
import pickle
import sys, os
from tqdm import tqdm
import torch
import numpy as np

CUDA = torch.cuda.is_available()

def parse_sentence(bert_vecs, mode:str="average") -> np.array:
    """
    Load an Bert embedder.
    :param elmo_vecs: the ELMo model results for a single sentence
    :param mode:
    :return:
    """
    if mode == "average":
        return np.average(bert_vecs, 0)
    elif mode == 'weighted_average':
        return np.swapaxes(bert_vecs, 0, 1)
    elif mode == 'last':
        return bert_vecs[-1, :, :]
    elif mode == 'all':
        return bert_vecs
    else:
        return bert_vecs



def load_bert(bert_path: str) -> DistilBertModel:
    """
    Load a Bert embedder
    :param bert_path:
    :return:
    """
    
    return transformers.DistilBertModel.from_pretrained(bert_path.split("/")[-1],)

class CustomDataset:
    def __init__(self, sentences, bert_path, padding = 140):
        self.sentences = sentences
        self.tokenizer = BertWordPieceTokenizer(f'{bert_path}/vocab.txt', lowercase=True)
        self.padding = padding

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        s = self.sentences[idx]#['[CLS]', *self.sentences[idx], '[SEP]']

        to_ignore_none = lambda x: x if x is not None else 0
        to_id = lambda x:to_ignore_none(self.tokenizer.token_to_id(x))
        
        n_pads = self.padding - len(s)
        x = list(map(to_id, s))
        assert(len(x) == len(s))
        x = x+[0 for _ in range(n_pads)]
        return torch.tensor(x), n_pads#, torch.tensor([])

def read_parse_write(bert: DistilBertModel, bert_path: str, infile: str, outfile: str, mode: str = "average", batch_size=0) -> None:
    """
    Read the input files and write the vectors to the output files
    :param bert: Bert embedder
    :param infile: input files for the sentences
    :param outfile: output vector files
    :param mode: the mode of elmo vectors
    :return:
    """
    reader = Reader()
    insts = reader.read_txt(infile, -1)
    f = open(outfile, 'wb')
    all_vecs = []
    all_sents = []
    for inst in insts:
        all_sents.append(inst.input.words)
    
    dataset = CustomDataset(all_sents,bert_path)

    batch_size = max(1, batch_size)# make sure batch_size is gt 0
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False, num_workers=4)
    for i, (batch, n_pads) in tqdm(enumerate(dataloader)):
        batch = batch.cuda() if CUDA else batch
        with torch.no_grad():
            bert_batch_vecs = bert(batch)[0].cpu().numpy()
            vectors = parse_sentence(bert_batch_vecs, mode=mode)
            for j in range(vectors.shape[0]):
                all_vecs.append(vectors[j,:-n_pads[j],:])

    print("Finishing embedding Bert sequences, saving the vector files.")
    pickle.dump(all_vecs, f)
    f.close()


def get_vector():
    dataset = sys.argv[1]
    DISTILBERT_PATH = f'data/{dataset}/distilbert-base-uncased'

    bert = load_bert(DISTILBERT_PATH)
    mode= "all"
    batch_size = 64 # >=1 for using batch-based inference


    # Read train
    file = "data/"+dataset+"/train.txt"
    outfile = file + ".bert.vec"
    read_parse_write(bert, DISTILBERT_PATH, file, outfile, mode, batch_size)

    # Read dev
    file = "data/"+dataset+"/dev.txt"
    outfile = file + ".bert.vec"
    read_parse_write(bert, DISTILBERT_PATH, file, outfile, mode, batch_size)

    # Read test
    file = "data/"+dataset+"/test.txt"
    outfile = file + ".bert.vec"
    read_parse_write(bert, DISTILBERT_PATH, file, outfile, mode, batch_size)



if __name__ == "__main__":
    get_vector()
