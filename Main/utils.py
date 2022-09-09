# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:41
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : utils.py
# @Software: PyCharm
# @Note    :
import json
import os
import shutil
import re
import time


def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)


def write_post(post_list, path):
    for post in post_list:
        write_json(post[1], os.path.join(path, f'{post[0]}.json'))


def dataset_makedirs(dataset_path):
    train_path = os.path.join(dataset_path, 'train', 'raw')
    val_path = os.path.join(dataset_path, 'val', 'raw')
    test_path = os.path.join(dataset_path, 'test', 'raw')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)
    os.makedirs(os.path.join(dataset_path, 'train', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'val', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'test', 'processed'))

    return train_path, val_path, test_path


def clean_comment(comment_text):
    match_res = re.match('回复@.*?:', comment_text)
    if match_res:
        return comment_text[len(match_res.group()):]
    else:
        return comment_text

def create_log_dict(args):
    log_dict = {}

    log_dict['dataset'] = args.dataset
    log_dict['runs'] = args.runs

    log_dict['batch_size'] = args.batch_size
    log_dict['epochs'] = args.epochs

    log_dict['k'] = args.k

    log_dict['emb_dim'] = args.emb_dim
    log_dict['num_structure_index'] = args.num_structure_index

    log_dict['include_key_structure'] = args.include_key_structure
    log_dict['include_val_structure'] = args.include_val_structure
    log_dict['word_module_version'] = args.word_module_version
    log_dict['post_module_version'] = args.post_module_version

    log_dict['train_word_emb'] = args.train_word_emb
    log_dict['train_pos_emb'] = args.train_pos_emb

    log_dict['size'] = args.size
    log_dict['include_time_interval'] = args.include_time_interval

    log_dict['d_model'] = args.d_model
    log_dict['dropout_rate'] = args.dropout_rate

    log_dict['ff_word'] = args.ff_word
    log_dict['num_emb_layers_word'] = args.num_emb_layers_word
    log_dict['n_mha_layers_word'] = args.n_mha_layers_word
    log_dict['n_head_word'] = args.n_head_word

    log_dict['ff_post'] = args.ff_post
    log_dict['num_emb_layers'] = args.num_emb_layers
    log_dict['n_mha_layers'] = args.n_mha_layers
    log_dict['n_head'] = args.n_head

    log_dict['d_feed_forward'] = args.d_feed_forward

    log_dict['learning_rate'] = args.learning_rate
    log_dict['beta_1'] = args.beta_1
    log_dict['beta_2'] = args.beta_2
    log_dict['n_warmup_steps'] = args.n_warmup_steps
    log_dict['vary_lr'] = args.vary_lr

    log_dict['record'] = []
    return log_dict


def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()


def str2timestamp(time_string):
    t = time.strptime(time_string, "%y-%m-%d %H:%M")
    timestamp = int(time.mktime(t))
    return timestamp


