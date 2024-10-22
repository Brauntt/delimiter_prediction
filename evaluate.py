import torch
import numpy as np
import time

from parser import args
from torch.autograd import Variable
from utils import subsequent_mask
from desc_tokenizer import feature_tokenize

def log(data, timestamp):
    file = open(f'log/log-{timestamp}.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()

def greedy_decode(model, src, src_mask, start_symbol):
    """
    使用模型生成概率序列
    """
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
    """
    根据预测结果对每一次翻译进行评价
    """
    ground_truth = ground_truth[1:len(ground_truth)-1]
    length = min(len(ground_truth),len(pre))
    TP = 0 #true positive
    FP = 0 #false positive
    TN = 0 #true negative
    FN = 0 #false negative
    for i in range(length):
        if ground_truth[i] == '2' and pre[i] == '2':
            TP = TP + 1
        if ground_truth[i] == '2' and pre[i] == '0':
            FN = FN + 1
        if ground_truth[i] == '0' and pre[i] == '2':
            FP = FP + 1
        if ground_truth[i] == '0' and pre[i] == '0':
            TN = TN + 1
    return TP,FP,TN,FN

def predict(data, model):
    """
    用户输入一条描述，使用现存模型对描述进行翻译
    """
    des = input()
    des = des + ' '
    tokenize_des = feature_tokenize(des)
    tokenize_des = ["BOS"] + tokenize_des + ["EOS"]
    print(" ".join(tokenize_des))
    src_index = [data.en_word_dict[w] for w in tokenize_des]
    src = torch.from_numpy(np.array(src_index)).long().to(args.device)
    src = src.unsqueeze(0)
    src_mask = (src != 0).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, start_symbol=data.cn_word_dict["BOS"])
    translation = []
    for j in range(1, out.size(1)):
        sym = data.cn_index_dict[out[0, j].item()]
        if sym != 'EOS':
            translation.append(sym)
        else:
            break
    print("pre %s" % " ".join(translation))
    return translation

def evaluate(data, model):
    """
    对完整测试数据集评价
    """
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
       
            

