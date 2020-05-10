import collections
import os
import re

import tensorflow
import keras

import approximateMatch
from model import *
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from datahelper import load_data,one_hot,load_vocab,load_char_vocab,load_char


def get_normal_F1(path="./output/predict.tsv"):
    # tP,tN,fP,fN = 0,0,0,0
    f1_score = np.zeros(shape=(3,3))
    f = open(path,"r",encoding="UTF-8")
    for l in f.readlines():
        if "\n" == l[0]:
            break
        if re.match(r"^EOS$",l) or re.match("^BOS$",l):
            continue
        line = l.split("\t")
        w,a,p = line[0],line[1],line[2]
        if a[0] == "B" and p[0] == "B":
            f1_score[0,0] += 1
        elif a[0] == "B" and p[0] == "I":
            f1_score[1,0] += 1
        elif a[0] == "B" and p[0] == "O":
            f1_score[2,0] += 1
        elif a[0] == "I" and p[0] == "B":
            f1_score[0,1] += 1
        elif a[0] == "I" and p[0] == "I":
            f1_score[1,1] += 1
        elif a[0] == "I" and p[0] == "O":
            f1_score[2,1] += 1
        elif a[0] == "O" and p[0] == "B":
            f1_score[0,2] += 1
        elif a[0] == "O" and p[0] == "I":
            f1_score[1,2] += 1
        elif a[0] == "O" and p[0] == "O":
            f1_score[2,2] += 1

    p_B = f1_score[0,0] / np.sum(f1_score[0,:])
    r_B = f1_score[0,0] / np.sum(f1_score[:,0])
    f_B = 2 * p_B * r_B / (p_B + r_B)
    p_I = f1_score[1,1] / np.sum(f1_score[1,:])
    r_I = f1_score[1,1] / np.sum(f1_score[:,1])
    f_I = 2 * p_I * r_I / (p_I + r_I)
    p_O = f1_score[2,2] / np.sum(f1_score[2,:])
    r_O = f1_score[2,2] / np.sum(f1_score[:,2])
    f_O = 2 * p_O * r_O / (p_O + r_O)
    p = (p_B + p_I + p_O) / 3
    r = (r_B + r_I + r_O) / 3
    f1 = (f_B + f_I + f_O) / 3

    print("f1,p,r: " + str(f1) + " " + str(p) + " " + str(r))
    f.close()

    f = open(path,"a",encoding="UTF-8")
    f.write("\nf1,p,r: " + str(f1) + " " + str(p) + " " + str(r))
    f.close()



def get_score(word,act,pre,vocab,tag,path="./output/predict.tsv"):
    pre = np.argmax(pre,axis=2)

    vocab = {str(v):k for k,v in vocab.items()}
    tag = {str(v):k for k,v in tag.items()}
    out = open(path,"w",encoding="UTF-8")
    for i,s in enumerate(word):
        out.write("BOS\tO\tO\n")
        for j,w in enumerate(s):
            w = vocab[str(w)]
            if "B" in tag[str(int(act[i][j]))]:
                a = "B-ADR"
            elif "I" in tag[str(int(act[i][j]))]:
                a = "I-ADR"
            elif "O" in tag[str(int(act[i][j]))]:
                a = "O"
            elif "pad" in tag[str(int(act[i][j]))]:
                a = "O"
            try:
                if "B" in tag[str(int(pre[i,j]))]:
                    p = "B-ADR"
                elif "I" in tag[str(int(pre[i,j]))]:
                    p = "I-ADR"
                elif "O" in tag[str(int(pre[i,j]))]:
                    p = "O"
                elif "pad" in tag[str(int(pre[i,j]))]:
                    p = "O"
            except IndexError as IE:
                p = "O"
            out.write("\t".join([w,a,p]) +"\n")
        out.write("EOS\tO\tO\n")
    scores = approximateMatch.get_approx_match(path)
    out.write("\nTEST Approximate Matching Results:\n  ADR: Precision "+ str(scores["p"]) + " Recall " + str(scores["r"]) + " F1 " + str(scores["f1"]))
    out.close()
    return scores


# load data

# train_path = "./data/NCBI-disease/train_dev.tsv"
# test_path = "./data/NCBI-disease/test.tsv"
# train_path = "./data/s800/train_dev.tsv"
# test_path = "./data/s800/test.tsv"
# train_path = "./data/BC2GM/train_dev.tsv"
# test_path = "./data/BC2GM/test.tsv"

train_path = "./data/Genia4ERtask1.txt"
test_path = "./data/Genia4EReval1.txt"
train = load_data(train_path)
test = load_data(test_path)
vocab,tag = load_vocab([train_path,test_path])
vocab["<pad>"] = 0
tag["<pad>"] = 0

maxlenth = 50
vocab_size = len(vocab)
tag_size = len(tag)  
print("vocab_size: ",vocab_size)


# char embedding
print("char embedding...")
maxlenth_char = 28
charembedding_size = 200
charvocab = load_char_vocab(train,test)
charvocab_size = len(charvocab)
train_char = load_char(train)
test_char = load_char(test)

def char_padding(chars,charvocab,maxlenth):
    for i,s in enumerate(chars):
        for j,w in enumerate(s):
            for k,c in enumerate(w):
                chars[i][j][k] = charvocab[c]

    pad_chars = []
    for i,s in enumerate(chars):
        while len(s) < maxlenth:
            s.append([])
        pad_chars.append(pad_sequences(s,maxlen=maxlenth_char,padding="post"))
    pad_chars = np.array(pad_chars)

    chars_shape = (pad_chars.shape[0],pad_chars[0].shape[0],pad_chars[0].shape[1])
    chars = np.zeros(shape=chars_shape,)
    for i in range(chars_shape[0]):
        for j in range(chars_shape[1]):
            for k in range(chars_shape[2]):
                chars[i,j,k] = pad_chars[i][j,k]
    return chars
train_char = char_padding(train_char,charvocab,maxlenth)
test_char = char_padding(test_char,charvocab,maxlenth)

# word2idx
train_token = one_hot(train["token"],vocab)
train_label = one_hot(train["label"],tag)
test_token = one_hot(test["token"],vocab)
test_label = one_hot(test["label"],tag)

# padding
train_input = pad_sequences(train_token,maxlen=maxlenth,padding="post")
train_label = pad_sequences(train_label,maxlen=maxlenth,padding="post")
test_input = pad_sequences(test_token,maxlen=maxlenth,padding="post")
test_label = pad_sequences(test_label,maxlen=maxlenth,padding="post")


# label processing
def label2vec(label,cls=4):
    vec = np.zeros((label.shape[0],label.shape[1],cls))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            t = int(label[i,j])
            vec[i,j,t] = 1
    return vec


train_label = label2vec(train_label,4)
test_label = label2vec(test_label,4)


# embedding
print("loading embedding...")
embedding_size = 200
w2v_dir_path = "/media/network/watching_dog/embedding/bio_nlp_vec/PubMed-shuffle-win-30.bin"
word2vec = KeyedVectors.load_word2vec_format(w2v_dir_path, binary=True, unicode_errors='ignore')

print("build embedding weights...")
embedding_weights = np.zeros((vocab_size,embedding_size))
unknow_words = []
know_words = []
for word, index in vocab.items():
    try:
        embedding_weights[index, :] = word2vec[word.lower()]
        know_words.append(word)
    except KeyError as E:
        # print(E)
        unknow_words.append(word)
        embedding_weights[index, :] = np.random.uniform(-0.025, 0.025, embedding_size)
print("unknow_per: ",len(unknow_words)/vocab_size," unkownwords: ",len(unknow_words)," vocab_size: ",vocab_size)


# model
print("model...")
model = myModel(maxSeqLenth=maxlenth,
                maxCharLenth=maxlenth_char,
                embeddingDim=embedding_size,
                charEmbeddingDim=charembedding_size,
                weight=embedding_weights,
                vocabSize=vocab_size,
                charvocabSize=charvocab_size,
                target=tag_size,
                path="./output/model_deep.h5",
                mask=False,
                )



model.cnn_rnn_attn(hiddenDim=400,
                   )


train_x = [train_input,train_char,]
train_y = train_label
test_x = [test_input,test_char,]


# train
print("train...")
epoch = 3

model.train_model(x=train_x,y=train_y,
                  epoch=epoch,
                  batch_size=16,
                  validation_split=0.1,
                  )
print("predict...")
test_predict = model.predict(test_x)
print("eval...")

result_path = "./output/model_epoch"+str(epoch)+".tsv"

score = get_score(test["token"],test["label"],test_predict,vocab,tag,path=result_path)
get_normal_F1(result_path)
print("match_F1: "+str(score["f1"]))


