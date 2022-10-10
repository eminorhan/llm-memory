import json, jsonlines
import numpy as np


def sample_indices(n, t):
       chosen = np.random.choice(n, size=2*t, replace=False)
       seen = chosen[:t]
       unseen = chosen[t:]
       return seen, unseen

def generate_expt_data(filename, t, seed, length_check=True):

       # set seed
       np.random.seed(seed)

       # read global data file into a list
       f = open(filename, 'r')
       x = []

       for line in f:
              x.append(json.loads(line))

       # sample seen-unseen indices
       n = len(x)
       seen, unseen = sample_indices(n, t)

       # retrieve corresponding data entries
       seen_data = [x[i] for i in seen]
       unseen_data = [x[i] for i in unseen]

       # sanity check: make sure seen-unseen lengths are roughly similar
       if length_check:
              seen_lengths = [len(d["sent"]) for d in seen_data]
              unseen_lengths = [len(d["sent"]) for d in unseen_data]
              print('Mean, std of seen length (chars):', np.mean(seen_lengths), np.std(seen_lengths))
              print('Mean, std of unseen length (chars):', np.mean(unseen_lengths), np.std(unseen_lengths))

       # write sampled seen-unseen data to jsonl files
       with jsonlines.open('seen_data_{}.jsonl'.format(seed), mode='w') as writer:
              writer.write_all(seen_data)

       with jsonlines.open('unseen_data_{}.jsonl'.format(seed), mode='w') as writer:
              writer.write_all(unseen_data)

# generate a bunch of experimental seen-unseen datasets
generate_expt_data('data.json', 600, 0)
generate_expt_data('data.json', 600, 1)
generate_expt_data('data.json', 600, 2)
generate_expt_data('data.json', 600, 3)