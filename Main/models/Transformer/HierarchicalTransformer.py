import torch
import torch.nn as nn
import torch.nn.functional as F
from Main.models.Transformer import Transformer
from Main.models.SubModules import WordModule
from Main.models.SubModules import PostModule

__author__ = "Serena Khoo"


class HierarchicalTransformer(nn.Module):

    @staticmethod
    def init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)

    def __init__(self, config):

        super(HierarchicalTransformer, self).__init__()

        # <----------- Config ----------->
        self.config = config

        # <----------- Both word and post modules ----------->
        if self.config.hitplan:
            self.word_module = WordModule.WordModule(config)
        self.post_module = PostModule.PostModule(config)

    def forward(self, X, word_pos, time_delay, structure, attention_mask_word=None, attention_mask_post=None,
                return_attention=False):
        # <----------- Passing through word module ----------->
        if self.config.hitplan:
            batch_size, num_posts, num_words, emb_dim = X.shape
            X_word, self_atten_weights_dict_word = self.word_module(X, word_pos, attention_mask=attention_mask_word)
        else:
            batch_size, num_posts, emb_dim = X.shape
            X_word = X

        # <----------- Passing through post module ----------->
        output, self_atten_output_post, self_atten_weights_dict_post = self.post_module(X_word, time_delay, batch_size,
                                                                                        num_posts, emb_dim,
                                                                                        structure=structure,
                                                                                        attention_mask=attention_mask_post)

        # <--------- Clear the memory -------------->
        torch.cuda.empty_cache()

        if return_attention:
            return output, self_atten_output_post, self_atten_weights_dict_word, self_atten_weights_dict_post

        # <-------- Delete the attention weights if not returning it ---------->
        if self.config.hitplan:
            del self_atten_weights_dict_word
        del self_atten_weights_dict_post
        del self_atten_output_post
        torch.cuda.empty_cache()

        return output
