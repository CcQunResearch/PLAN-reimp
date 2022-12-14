# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 10:35
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import os.path as osp
import json
import numpy as np
import torch
from torchtext.legacy.data import Field, NestedField, Example, Dataset, BucketIterator
from torchtext import vocab
from Main.utils import str2timestamp
import spacy
from spacy.lang.zh import Chinese

nlp = Chinese()


class WeiboDataLoader():

    def __init__(self, root, word_embedding_dir, word_embedding_filename, max_length,
                 max_tweet, num_structure_index, num_bin, batch_size, clean=True):
        self.root = root
        self.word_embedding_dir = word_embedding_dir
        self.word_embedding_filename = word_embedding_filename
        self.num_structure_index = num_structure_index
        self.max_length = max_length
        self.max_tweet = max_tweet
        self.num_bin = num_bin
        self.batch_size = batch_size
        self.clean = clean
        self.run_pipeline()

    def get_data(self, dataset_type):
        assert dataset_type in ["train", "test", "val"]

        data = self.train_batch if dataset_type == "train" else self.test_batch if dataset_type == "test" else self.val_batch if dataset_type == "val" else "something wrong"

        for batch in data:
            X = getattr(batch, 'tweets')
            y = getattr(batch, 'label')
            structure = getattr(batch, 'structure')
            time_delay = getattr(batch, 'time_delay')

            num_events, num_posts, num_words, = X.shape

            word_pos = np.repeat(
                np.expand_dims(np.repeat(np.expand_dims(np.arange(num_words), axis=0), num_posts, axis=0), axis=0),
                num_events, axis=0)
            word_pos = torch.from_numpy(word_pos)

            attention_mask_word = torch.where((X == 1), torch.zeros(1), torch.ones(1)).type(torch.FloatTensor)
            check = torch.sum(torch.where((X == 1), torch.ones(1), torch.zeros(1)), dim=-1)
            attention_mask_post = torch.where((check == num_words), torch.zeros(1), torch.ones(1)).type(
                torch.FloatTensor)

            yield X, y, word_pos, time_delay, structure, attention_mask_word, attention_mask_post

    @staticmethod
    def tokenize_structure(structure_lst):
        return structure_lst

    def tokenize_text(self, text):
        if self.max_length == 0:
            token_lst = [token.text for token in nlp(text)]
        else:
            token_lst = [token.text for token in nlp(text)][:self.max_length]
        return token_lst

    # Step 1: Define the data fields
    def define_fields(self):
        self.tweet_field = Field(sequential=True,
                                 tokenize=self.tokenize_text)

        self.timestamp_field = Field(sequential=False,
                                     use_vocab=False)

        self.label_field = Field(sequential=False,
                                 use_vocab=False)

        self.structure_field = Field(sequential=True,
                                     tokenize=lambda x: WeiboDataLoader.tokenize_structure(x),
                                     pad_token=self.num_structure_index,
                                     use_vocab=False)

        self.tweet_lst_field = NestedField(self.tweet_field)

        self.timestamp_lst_field = NestedField(self.timestamp_field,
                                               pad_token=str(self.num_bin))

        self.structure_lst_field = NestedField(self.structure_field)

        data_fields = [('tweets', self.tweet_lst_field), ('time_delay', self.timestamp_lst_field),
                       ('structure', self.structure_lst_field), ('label', self.label_field)]
        self.data_fields = data_fields

    # Step 2: Reading the data
    def read_data(self):
        examples = []
        raw_dir = osp.join(self.root, 'raw')
        raw_file_names = os.listdir(raw_dir)

        if self.clean:
            limit_num = self.max_tweet
            pass_comment = ['', '????????????', '???????????????', '????????????', '???????????????']
            for filename in raw_file_names:
                filepath = osp.join(raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                pass_num = 0
                id_to_index = {}
                del_index = []
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        del_index.append(i)
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    if comment['parent'] != -1:
                        comment['parent'] = post['comment'][id_to_index[comment['parent']]]['comment id']
                comments = [comment for i, comment in enumerate(post['comment'][:limit_num]) if i not in del_index]
                source_comment = {
                    'comment id': -1,
                    'parent': None,
                    'content': post['source']['content'],
                    'time': post['source']['time']
                }
                comments = [source_comment] + comments
                for i in range(len(comments)):
                    comments[i]['comment id'] += 1
                    if comments[i]['parent'] != None:
                        comments[i]['parent'] += 1
                    comments[i]['time'] = str2timestamp(comments[i]['time'])

                # if self.max_tweet > 0:
                #     comments = comments[:self.max_tweet]

                tweets = []
                time_delay = []
                structure = []
                label = post['source']['label']
                for i in range(len(comments)):
                    tweets.append("".join(comments[i]['content'].split()))
                    bin = int((comments[i]['time'] - comments[0]['time']) / 600)
                    bin = bin if bin < self.num_bin else self.num_bin - 1
                    time_delay.append(bin)
                    node_structure = []
                    for j in range(len(comments)):
                        if i == j:
                            node_structure.append(4)
                            continue
                        elif comments[i]['parent'] == j:
                            node_structure.append(0)
                            continue
                        elif comments[j]['parent'] == i:
                            node_structure.append(1)
                            continue
                        elif comments[i]['time'] > comments[j]['time']:
                            node_structure.append(2)
                            continue
                        elif comments[i]['time'] < comments[j]['time']:
                            node_structure.append(3)
                            continue
                        else:
                            rel = 3 if i < j else 2
                            node_structure.append(rel)
                            continue

                    structure.append(node_structure)
                examples.append(Example.fromlist([tweets, time_delay, structure, label], self.data_fields))
        else:
            for filename in raw_file_names:
                filepath = osp.join(raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                comments = post['comment']
                source_comment = {
                    'comment id': -1,
                    'parent': None,
                    'content': post['source']['content'],
                    'time': post['source']['time']
                }
                comments = [source_comment] + comments
                for i in range(len(comments)):
                    comments[i]['comment id'] += 1
                    if comments[i]['parent'] != None:
                        comments[i]['parent'] += 1
                    comments[i]['time'] = str2timestamp(comments[i]['time'])

                if self.max_tweet > 0:
                    comments = comments[:self.max_tweet]

                tweets = []
                time_delay = []
                structure = []
                label = post['source']['label']
                for i in range(len(comments)):
                    tweets.append("".join(comments[i]['content'].split()))
                    bin = int((comments[i]['time'] - comments[0]['time']) / 600)
                    bin = bin if bin < self.num_bin else self.num_bin - 1
                    time_delay.append(bin)
                    node_structure = []
                    for j in range(len(comments)):
                        if i == j:
                            node_structure.append(4)
                            continue
                        elif comments[i]['parent'] == j:
                            node_structure.append(0)
                            continue
                        elif comments[j]['parent'] == i:
                            node_structure.append(1)
                            continue
                        elif comments[i]['time'] > comments[j]['time']:
                            node_structure.append(2)
                            continue
                        elif comments[i]['time'] < comments[j]['time']:
                            node_structure.append(3)
                            continue
                        else:
                            rel = 3 if i < j else 2
                            node_structure.append(rel)
                            continue
                    structure.append(node_structure)
                examples.append(Example.fromlist([tweets, time_delay, structure, label], self.data_fields))

        self.examples = examples

    # Step 3: Building the vectors
    def build_vectors(self):
        full_dataset = Dataset(self.examples, self.data_fields)
        vec = vocab.Vectors(name=self.word_embedding_filename, cache=self.word_embedding_dir)
        self.tweet_field.build_vocab(getattr(full_dataset, 'tweets'), min_freq=5, vectors=vec)

    def run_pipeline(self):
        # Step 1 : Define the data fields
        self.define_fields()

        # Step 2: Reading the data
        self.read_data()

        # Step 3: Building the vectors
        self.build_vectors()

    def load_batches(self, dataset, batch_size=0):
        bs = batch_size if batch_size != 0 else self.batch_size
        batch = BucketIterator(dataset=dataset,
                               batch_size=bs,
                               sort_key=lambda x: len(getattr(x, 'tweets')),
                               sort_within_batch=True,
                               repeat=False)
        return batch

    def split_dataset(self, train_index, test_index, val_index):
        train_examples = [example for i, example in enumerate(self.examples) if i in train_index]
        test_examples = [example for i, example in enumerate(self.examples) if i in test_index]
        val_examples = [example for i, example in enumerate(self.examples) if i in val_index]

        self.train_dataset = Dataset(train_examples, self.data_fields)
        self.test_dataset = Dataset(test_examples, self.data_fields)
        self.val_dataset = Dataset(val_examples, self.data_fields)

        self.train_batch = self.load_batches(self.train_dataset)
        self.test_batch = self.load_batches(self.test_dataset, 1)
        self.val_batch = self.load_batches(self.val_dataset, 1)
