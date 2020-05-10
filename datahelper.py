import collections
import re
import os
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_char_vocab(train,test):
    char_per_word = []
    char_word = []
    chars = []
    maxlen_char_word = 0
    over_char = []
    maxlenth_char = 28

    for s in (train["token"] + test["token"]):
        for w in s:
            for c in w.lower():
                char_per_word.append(c)

            if len(char_per_word) > maxlenth_char:
                over_char.append(char_per_word)
                char_per_word = char_per_word[:maxlenth_char]
            if len(char_per_word) > maxlen_char_word:
                maxlen_char_word = len(char_per_word)

            char_word.append(char_per_word)
            char_per_word = []

        chars.append(char_word)
        char_word = []

    charcounts = Counter()
    for senc in chars:
        for word in senc:
            for charac in word:
                charcounts[charac] += 1
    charvocab = [charcount[0] for charcount in charcounts.most_common()]
    charvocab = {c: i+1 for i, c in enumerate(charvocab)}
    charvocab["<pad>"] = 0
    return charvocab

def load_char(data):
    char_per_word = []
    char_word = []
    chars = []
    maxlen_char_word = 0
    over_char = []
    maxlenth_char = 28

    for s in (data["token"]):
        for w in s:
            for c in w.lower():
                char_per_word.append(c)

            if len(char_per_word) > maxlenth_char:
                over_char.append(char_per_word)
                char_per_word = char_per_word[:maxlenth_char]
            if len(char_per_word) > maxlen_char_word:
                maxlen_char_word = len(char_per_word)

            char_word.append(char_per_word)
            char_per_word = []

        chars.append(char_word)
        char_word = []
    return chars

def one_hot(tokens,vocab):
    for i,seq in enumerate(tokens):
        for j,t in enumerate(seq):
            tokens[i][j] = vocab[t]
    return tokens

def token_cleaner(token):
    # token = ""
    if token.isdigit():
        return "<num>"
    else:
        return token.lower()

def load_vocab(path,mask=True):
    counter = Counter()
    tag = Counter()
    for p in path:
        for l in open(p,"r",encoding="UTF-8").readlines():
            line = l.split()
            try:
                if line[0].isdigit():
                    counter["<num>"] += 1
                else:
                    counter[line[0].lower()] += 1
                tag[line[1]] += 1
            except Exception:
                pass
    counter = sorted(counter.most_common(),key=lambda x:x[1],reverse=True)
    tag = sorted(tag.most_common(),key=lambda x:x[1],reverse=True)
    if mask:
        return {v[0]:i+1 for i,v in enumerate(counter)},\
               {v[0]:i+1 for i,v in enumerate(tag)}  # 0 as mask
    else:
        return {v[0]:i for i,v in enumerate(counter)}, \
               {v[0]:i for i,v in enumerate(tag)}  # 0 as mask


def load_data(path):
    data = open(path,"r",encoding="UTF-8")
    counter = Counter()
    tags = Counter()

    # load tokens and labels
    tokens = []
    labels = []
    token_seq = []
    label_seq = []
    for l in data.readlines():
        line = l.split()
        try:
            token,label = token_cleaner(line[0]),line[1]
            # counter[token] += 1
            # tags[label] += 1
            token_seq.append(token)
            label_seq.append(label)
        except IndexError as IE:
            tokens.append(token_seq)
            labels.append(label_seq)
            token_seq = []
            label_seq = []

    # load vocab
    # vocab = load_vocab(counter)
    # tag = load_vocab(tags)

    # label processing
    # for i,s in enumerate(labels):
    #     for j,t in enumerate(s):
    #         labels[i][j] = [1 if _ == int(t) else 0 for _ in range(len(tag))]
    # print(vocab)
    # print(tag)
    # print(labels[0])
    return {"token":tokens,"label":labels,}


if __name__ == "__main__":
    # data = load_data("./data/Genia4ERtask1.txt")
    # counter = Counter()
    # for s in data[0]:
    #     counter[str(len(s))] += 1
    # counter = sorted(counter.most_common(),key=lambda x:x[1])
    # y = [int(i[0]) for i in counter]
    # x = [i[1] for i in counter]
    # print(max(x))
    # print(max(y))
    # plt.plot(x,y)
    # plt.show()
    #
    # c = 0
    # for _ in data[0]:
    #     if len(_) > 60:
    #         c += 1
    # print(c/len(data[0])*100)
    train_path = "./data/Genia4ERtask1.txt"
    test_path = "./data/Genia4EReval1.txt"
    train = load_data(train_path)
    test = load_data(test_path)
    maxlenWord = ""
    words = []
    print("check...")
    for s in train["token"] + test["token"]:
        for w in s:
            if len(w) > len(maxlenWord):
                # print(w)
                maxlenWord = w
            if w not in words:
                words.append(w)
    print(len(maxlenWord)," : ",maxlenWord) # maxlenth = 28
    # words = sorted(words,reverse=True)
    # lenWord = [len(w) for w in words]
    # print(np.mean(lenWord))
    # print(np.median(lenWord))
    # words = sorted(words)
    check = [w for w in words if len(w) <= 1 and w not in "qwertyuioplkjhgfdsazxcvbnm"]
    print(len(check)," : ",check)  # len(check) = 21
