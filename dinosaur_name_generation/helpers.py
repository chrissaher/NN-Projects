import numpy as np

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
smooth = lambda loss, cur_loss: loss * 0.999 + cur_loss * 0.001
get_initial_loss = lambda vocab_size, seq_length: -np.log(1.0/vocab_size)*seq_length

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    # for python 3.x
    print('%s'%(txt, ), end='')
    # for python 2.x
    # print txt,
