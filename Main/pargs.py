import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Weibo')
    parser.add_argument('--runs', type=int, default=10)
    # 'zh' for chinese, 'en' for english
    # parser.add_argument('--language', type=str, default='zh')

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--k', type=int, default=10000)

    # Word padding settings
    parser.add_argument('--max_length', type=int, default=0)  # Pad the content to max_length words

    # Word Embedding settings
    parser.add_argument('--pos_num', type=int, default=500)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--num_structure_index', type=int, default=5)

    # Versioning of methods
    parser.add_argument('--include_key_structure', type=str2bool, default=True)
    parser.add_argument('--include_val_structure', type=str2bool, default=True)
    # {0: max_pooling, 1: average_pooling, 2: max_pooling_w_attention, 3: average_pooling_w_attention, 4: attention}
    parser.add_argument('--word_module_version', type=int, default=4)
    # {0: average_pooling, 1: condense_into_fix_vector, 2: first_vector, 3: attention}
    parser.add_argument('--post_module_version', type=int, default=3)

    # Word Embedding training
    parser.add_argument('--train_word_emb', type=str2bool, default=False)
    parser.add_argument('--train_pos_emb', type=str2bool, default=False)

    # Time interval embedding
    parser.add_argument('--size', type=int, default=100)  # Number of bins
    parser.add_argument('--include_time_interval', type=str2bool, default=True)

    # Model parameters settings
    parser.add_argument('--d_model', type=int, default=300)
    parser.add_argument('--dropout_rate', type=float, default=0.3)

    # <------------------------ WORD LEVEL ------------------------>
    parser.add_argument('--ff_word', type=str2bool, default=True)
    # Model parameters settings (To encode query, key and val)
    parser.add_argument('--num_emb_layers_word', type=int, default=2)
    # Number of Multihead Attention layers
    parser.add_argument('--n_mha_layers_word', type=int, default=2)
    # Number of MHA heads
    parser.add_argument('--n_head_word', type=int, default=2)

    # <------------------------ POST LEVEL ------------------------>
    parser.add_argument('--ff_post', type=str2bool, default=True)
    # Model parameters settings (To encode query, key and val)
    parser.add_argument('--num_emb_layers', type=int, default=2)
    # Number of Multihead Attention layers
    parser.add_argument('--n_mha_layers', type=int, default=12)
    # Number of MHA heads
    parser.add_argument('--n_head', type=int, default=2)

    # Model parameters settings (For feedforward network)
    parser.add_argument('--d_feed_forward', type=int, default=600)

    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--beta_1', type=float, default=0.90)
    parser.add_argument('--beta_2', type=float, default=0.98)
    parser.add_argument('--n_warmup_steps', type=int, default=6000)
    parser.add_argument('--vary_lr', type=str2bool, default=True)

    args = parser.parse_args()
    return args
