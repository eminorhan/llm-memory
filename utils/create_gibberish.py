import jsonlines
import random

# read and save file names
read_fname = 'seen_data_3.jsonl'
save_fname = 'unseen_data_3.jsonl'

data = []

with jsonlines.open(read_fname) as reader:
    for obj in reader:
      sent = obj['sent']
      words = sent.split()
      shuffled_words = []

      # shuffle letters within each word
      for word in words:
        shuffled = list(word)
        random.shuffle(shuffled)
        shuffled = "".join(shuffled)
        shuffled_words.append(shuffled)

      # shuffle words within sentence
      random.shuffle(shuffled_words)  
      new_sent = " ".join(shuffled_words)
      data.append({"sent": new_sent, "sent_id": obj['sent_id'], "url_id": 0})

# save synonyms to file
with jsonlines.open(save_fname, mode='w') as writer:
       writer.write_all(data)