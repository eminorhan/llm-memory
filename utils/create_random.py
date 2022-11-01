import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import jsonlines
import random

moby = set(nltk.Text(gutenberg.words('melville-moby_dick.txt')))
moby = [word.lower() for word in moby if len(word)>2]

num_sent = 600
len_sent = 25
num_files = 8

for f in range(num_files):

  data = []
  for s in range(num_sent):
    sent = random.choices(moby, k=len_sent)
    sent = " ".join(sent)
    data.append({"sent": sent, "sent_id": s})
    print(f, s, sent)

  # write to jsonl file
  with jsonlines.open('random_{}.jsonl'.format(f), mode='w') as writer:
        writer.write_all(data)