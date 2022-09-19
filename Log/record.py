# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 20:26
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : record.py
# @Software: PyCharm
# @Note    :
import os
import sys
import json
import math
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))

cal_mean = -10

if __name__ == '__main__':
    log_dir_path = os.path.join(dirname, '..', 'Log')

    for filename in os.listdir(log_dir_path):
        if filename[-4:] == 'json':
            print(f'【{filename[:-5]}】')
            filepath = os.path.join(log_dir_path, filename)

            log = json.load(open(filepath, 'r', encoding='utf-8'))
            print('dataset', log['dataset'])
            print('runs', log['runs'])

            print('hitplan', log['hitplan'])
            print('batch_size', log['batch_size'])
            print('epochs', log['epochs'])

            print('max_length', log['max_length'])
            print('max_tweet', log['max_tweet'])

            print('emb_dim', log['emb_dim'])
            print('num_structure_index', log['num_structure_index'])

            print('include_key_structure', log['include_key_structure'])
            print('include_val_structure', log['include_val_structure'])
            print('word_module_version', log['word_module_version'])
            print('post_module_version', log['post_module_version'])

            print('train_word_emb', log['train_word_emb'])
            print('train_pos_emb', log['train_pos_emb'])

            print('size', log['size'])
            print('include_time_interval', log['include_time_interval'])

            print('d_model', log['d_model'])
            print('dropout_rate', log['dropout_rate'])

            print('ff_word', log['ff_word'])
            print('num_emb_layers_word', log['num_emb_layers_word'])
            print('n_mha_layers_word', log['n_mha_layers_word'])
            print('n_head_word', log['n_head_word'])

            print('ff_post', log['ff_post'])
            print('num_emb_layers', log['num_emb_layers'])
            print('n_mha_layers', log['n_mha_layers'])
            print('n_head', log['n_head'])

            print('d_feed_forward', log['d_feed_forward'])

            print('learning_rate', log['learning_rate'])
            print('beta_1', log['beta_1'])
            print('beta_2', log['beta_2'])
            print('n_warmup_steps', log['n_warmup_steps'])
            print('vary_lr', log['vary_lr'])

            acc_list = []
            for run in log['record']:
                # mean_acc = run['mean acc']
                mean_acc = round(np.mean(run['test accs'][cal_mean:]), 3)
                acc_list.append(mean_acc)

            mean = round(sum(acc_list) / len(acc_list), 3)
            sd = round(math.sqrt(sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)), 3)
            maxx = max(acc_list)
            print('test acc: {:.3f}±{:.3f}'.format(mean, sd))
            print('max acc: {:.3f}'.format(maxx))
            print()
