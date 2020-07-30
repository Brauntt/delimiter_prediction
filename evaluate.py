import torch
import numpy as np
import time

from parser import args
from torch.autograd import Variable
from utils import subsequent_mask

def log(data, timestamp):
    file = open(f'log/log-{timestamp}.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()

def greedy_decode(model, src, src_mask, start_symbol):
    memory = model.encode(src, src_mask)
    max_len = src.shape[1] - 1
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def label_precision(ground_truth, pre):
    ground_truth = ground_truth[1:len(ground_truth)-1]
    length = min(len(ground_truth),len(pre))
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    TPR  = 0
    FPR = 0
    for i in range(length):
        if ground_truth[i] == '2' and pre[i] == '2':
            TP = TP + 1
        if ground_truth[i] == '2' and pre[i] == '0':
            FN = FN + 1
        if ground_truth[i] == '0' and pre[i] == '2':
            FP = FP + 1
        if ground_truth[i] == '0' and pre[i] == '0':
            TN = TN + 1
    # if ((TP + FN) != 0):
    #     TPR = TP / (TP + FN)
    # if ((FP + TN) != 0):
    #     FPR = FP / (FP + TN)
    return TP,FP,TN,FN

def evaluate(data, model):
    timestamp = time.time()
    with torch.no_grad():
        precision = np.zeros((len(data.dev_en), 4))
        for i in range(len(data.dev_en)):
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            log(en_sent, timestamp)
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))
            log(cn_sent, timestamp)
            print(data.dev_en[i])
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(args.device)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)

            out = greedy_decode(model, src, src_mask, start_symbol = data.cn_word_dict["BOS"])

            translation = []
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    translation.append(sym)
                else:
                    break
            precision[i, 0],precision[i, 1],precision[i, 2],precision[i, 3] = label_precision([data.cn_index_dict[w] for w in data.dev_cn[i]],translation)
            # print(precision[i,0,0])
            # print(precision[i,0,1])
            print("pre %s" % " ".join(translation))
            log("translation: " + " ".join(translation) + "\n", timestamp)
    return precision
       
            

