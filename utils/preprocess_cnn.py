import jsonlines
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

split = "train"  # validation, test, train
cnn = load_dataset("cnn_dailymail", '3.0.0', split=split)

data = []
sent_id = 0
art_id = 0

for item in cnn:
       article = item['article']
       # remove short articles
       if len(article) > 100:
              sentences = sent_tokenize(article)
              for sent in sentences:
                     # remove short sentences
                     if len(sent) > 100:
                            data.append({"sent": sent, "sent_id": sent_id, "art_id": art_id})
                            sent_id += 1
                            if sent_id % 10000 == 0:
                                   print(sent_id)
                                   print(sent)
              art_id += 1

# write to jsonl file
with jsonlines.open('data_cnn_{}.jsonl'.format(split), mode='w') as writer:
       writer.write_all(data)