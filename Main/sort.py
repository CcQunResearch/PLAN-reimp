# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:40
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : sort.py
# @Software: PyCharm
# @Note    :
import os
import json
import random
import shutil
import time
from Main.utils import write_post, dataset_makedirs


def sort_weibo_dataset(source_path, dataset_path):
    post_id_list = []
    post_label_list = []
    all_post = []

    label_path = os.path.join(source_path, 'Weibo.txt')
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    raw_path = os.path.join(dataset_path, 'raw')
    processed_path = os.path.join(dataset_path, 'processed')
    os.makedirs(raw_path)
    os.makedirs(processed_path)

    f = open(label_path, 'r', encoding='utf-8')
    post_list = f.readlines()
    for post in post_list:
        post_id_list.append(post.split()[0].strip()[4:])
        post_label_list.append(int(post.split()[1].strip()[-1]))

    for i, post_id in enumerate(post_id_list):
        reverse_dict = {}
        comment_index = 0
        comment_list = []

        post_path = os.path.join(source_path, 'post', f'{post_id}.json')
        post = json.load(open(post_path, 'r', encoding='utf-8'))
        source = {
            'content': post[0]['text'],
            'user id': post[0]['uid'],
            'tweet id': post[0]['mid'],
            'label': post_label_list[i],
            'time': time.strftime("%y-%m-%d %H:%M", time.localtime(post[0]['t']))
        }

        for j in range(1, len(post)):
            comment_list.append({'comment id': comment_index, 'parent': -2, 'children': []})
            reverse_dict[post[j]['mid']] = comment_index
            comment_index += 1
        for k in range(1, len(post)):
            comment_list[k - 1]['content'] = post[k]['text']
            comment_list[k - 1]['user id'] = post[k]['uid']
            comment_list[k - 1]['user name'] = post[k]['username']
            comment_list[k - 1]['time'] = time.strftime("%y-%m-%d %H:%M", time.localtime(post[k]['t']))
            if post[k]['parent'] == source['tweet id']:
                comment_list[k - 1]['parent'] = -1
            else:
                parent_index = reverse_dict[post[k]['parent']]
                comment_list[k - 1]['parent'] = parent_index
                comment_list[parent_index]['children'].append(k - 1)
        all_post.append((post_id, {'source': source, 'comment': comment_list}))

    write_post(all_post, raw_path)


def sort_weibo_2class_dataset(label_source_path, label_dataset_path):
    if os.path.exists(label_dataset_path):
        shutil.rmtree(label_dataset_path)
    raw_path = os.path.join(label_dataset_path, 'raw')
    processed_path = os.path.join(label_dataset_path, 'processed')
    os.makedirs(raw_path)
    os.makedirs(processed_path)

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))

    write_post(all_post, raw_path)
