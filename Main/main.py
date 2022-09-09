# -*- coding: utf-8 -*-
# @Time    : 2022/9/6 19:56
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main.py
# @Software: PyCharm
# @Note    :
import sys
import os.path as osp
import warnings

dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))
warnings.filterwarnings("ignore")

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from Main.pargs import pargs
from Main.utils import create_log_dict, write_log, write_json
from Main.word2vec import Embedding, collect_sentences, train_word2vec, save_word_embedding
from Main.sort import sort_weibo_dataset, sort_weibo_self_dataset, sort_weibo_2class_dataset
from Main.dataset import WeiboDataLoader
from Main.models import WordEncoder, PositionEncoder, HierarchicalTransformer
from Main.optimizer import Optimizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(word_encoder, word_pos_encoder, time_delay_encoder, hierarchical_transformer, dataloader, device):
    hierarchical_transformer.train()
    total_loss = 0

    for X, y, word_pos, time_delay, structure, attention_mask_word, attention_mask_post in dataloader.get_data('train'):
        optimizer.zero_grad()

        X = word_encoder(X)
        word_pos = word_pos_encoder(word_pos)
        time_delay = time_delay_encoder(time_delay)

        X = X.to(device)
        y = y.to(device)
        word_pos = word_pos.to(device)
        time_delay = time_delay.to(device)
        structure = structure.to(device)
        attention_mask_word = attention_mask_word.to(device)
        attention_mask_post = attention_mask_post.to(device)
        pred = hierarchical_transformer(X, word_pos, time_delay, structure,
                                        attention_mask_word=attention_mask_word,
                                        attention_mask_post=attention_mask_post)

        y = y.to(torch.float32)
        loss = F.binary_cross_entropy(pred, y)
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item() * X.shape[0]
    return total_loss / len(dataloader.train_dataset)


def test(word_encoder, word_pos_encoder, time_delay_encoder, hierarchical_transformer, dataloader, dataset_type,
         device):
    hierarchical_transformer.eval()
    error = 0

    y_true = []
    y_pred = []
    for X, y, word_pos, time_delay, structure, attention_mask_word, attention_mask_post in dataloader.get_data(
            dataset_type):
        X = word_encoder(X)
        word_pos = word_pos_encoder(word_pos)
        time_delay = time_delay_encoder(time_delay)

        X = X.to(device)
        y = y.to(device)
        word_pos = word_pos.to(device)
        time_delay = time_delay.to(device)
        structure = structure.to(device)
        attention_mask_word = attention_mask_word.to(device)
        attention_mask_post = attention_mask_post.to(device)
        pred = hierarchical_transformer(X, word_pos, time_delay, structure,
                                        attention_mask_word=attention_mask_word,
                                        attention_mask_post=attention_mask_post)

        error += F.binary_cross_entropy(pred, y.to(torch.float32)).item() * X.shape[0]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        y_true += y.tolist()
        y_pred += pred.tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    length = len(dataloader.train_dataset) if dataset_type == 'train' else len(
        dataloader.test_dataset) if dataset_type == 'test' else len(
        dataloader.val_dataset) if dataset_type == "val" else "something wrong"
    return error / length, acc, prec, rec, f1


def test_and_log(word_encoder, word_pos_encoder, time_delay_encoder, hierarchical_transformer, dataloader,
                 device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(word_encoder, word_pos_encoder, time_delay_encoder,
                                                         hierarchical_transformer, dataloader, 'val', device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(word_encoder, word_pos_encoder, time_delay_encoder,
                                                              hierarchical_transformer, dataloader, 'test', device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    log_record['val accs'].append(round(val_acc, 3))
    log_record['test accs'].append(round(test_acc, 3))
    log_record['test prec T'].append(round(test_prec[0], 3))
    log_record['test prec F'].append(round(test_prec[1], 3))
    log_record['test rec T'].append(round(test_rec[0], 3))
    log_record['test rec F'].append(round(test_rec[1], 3))
    log_record['test f1 T'].append(round(test_f1[0], 3))
    log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, log_record


if __name__ == '__main__':
    args = pargs()

    dataset = args.dataset
    emb_dim = args.emb_dim
    device = torch.device(args.gpu if args.cuda else "cpu")
    # device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k
    epochs = args.epochs

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model', f'w2v_{dataset}_{emb_dim}.model')
    word_embedding_dir = osp.join(dirname, '..', 'Model')
    word_embedding_filename = f'w2v_{dataset}_{emb_dim}.txt'

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict(args)

    if not osp.exists(model_path):
        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path)
        elif dataset == 'Weibo-2class' or dataset == 'Weibo-2class-long':
            sort_weibo_2class_dataset(label_source_path, label_dataset_path)

        sentences = collect_sentences(label_dataset_path)
        w2v_model = train_word2vec(sentences, emb_dim)
        w2v_model.save(model_path)

    save_word_embedding(osp.join(word_embedding_dir, word_embedding_filename), model_path)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                      'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

        # if dataset == 'Weibo':
        #     sort_weibo_dataset(label_source_path, label_dataset_path, k)
        # elif dataset == 'Weibo-2class' or dataset == 'Weibo-2class-long':
        #     sort_weibo_2class_dataset(label_source_path, label_dataset_path, k)

        dataloader = WeiboDataLoader(label_dataset_path, word_embedding_dir, word_embedding_filename, args.max_length,
                                     args.num_structure_index, args.size, args.batch_size)

        word_encoder = WordEncoder(args, dataloader)
        word_pos_encoder = PositionEncoder(args, args.pos_num)
        time_delay_encoder = PositionEncoder(args, args.size)

        hierarchical_transformer = HierarchicalTransformer(args).to(device)

        adam_optimizer = Adam(hierarchical_transformer.parameters(),
                              np.power(args.d_model, - 0.5),
                              betas=(args.beta_1, args.beta_2))
        optimizer = Optimizer(args, adam_optimizer)

        val_error, log_info, log_record = test_and_log(word_encoder, word_pos_encoder, time_delay_encoder,
                                                       hierarchical_transformer, dataloader,
                                                       device, 0, adam_optimizer.param_groups[0]['lr'], 0, 0,
                                                       log_record)
        write_log(log, log_info)

        for epoch in range(1, epochs + 1):
            lr = optimizer.optimizer.param_groups[0]['lr']
            _ = train(word_encoder, word_pos_encoder, time_delay_encoder, hierarchical_transformer, dataloader, device)

            train_error, train_acc, _, _, _ = test(word_encoder, word_pos_encoder, time_delay_encoder,
                                                   hierarchical_transformer, dataloader, 'train', device)
            val_error, log_info, log_record = test_and_log(word_encoder, word_pos_encoder, time_delay_encoder,
                                                           hierarchical_transformer, dataloader, device, epoch,
                                                           lr, train_error, train_acc, log_record)
            write_log(log, log_info)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)