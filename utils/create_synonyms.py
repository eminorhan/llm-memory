import os
import time
import openai
import jsonlines

# replace with your own key
os.getenv("OPENAI_API_KEY")

# read and save file names
read_fname = 'data/seen_data_3.jsonl'
save_fname = 'data/synonym_seen_3.jsonl'

prefix = "Replace a single word in the given sentence by a synonym. Here are some examples to help guide you:\n\n" \
"Sentence: They’re irrational about what policies they favor.\n" \
"Replacement: They’re irrational about what policies they prefer.\n\n" \
"Sentence: It is up to us to judge what strikes us as the most plausible position.\n" \
"Replacement: It is up to us to evaluate what strikes us as the most plausible position.\n\n" \
"Sentence: Strong and stable leadership might change the pattern.\n" \
"Replacement: Strong and stable leadership might alter the pattern.\n\n" \
"Sentence: People foolishly assume that political and philosophical issues are simple.\n" \
"Replacement: People unwisely assume that political and philosophical issues are simple.\n\n" \
"Sentence: Feminists are correct to question our moral complacency.\n" \
"Replacement: Feminists are right to question our moral complacency.\n\n" \
"Sentence: "

postfix = "\nReplacement:"

data = []

with jsonlines.open(read_fname) as reader:
    for obj in reader:
      sent = obj['sent']
      prompt = prefix + sent + postfix

      response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=128,
        temperature=0.99
      )
      resp = response["choices"][0]["text"]
      resp = resp.replace("\n", "")
      print(sent)
      print(resp)
      print("\n")
      data.append({"sent": resp, "sent_id": obj['sent_id'], "url_id": obj['url_id']})
      time.sleep(1)  # to prevent rate limit error

# save synonyms to file
with jsonlines.open(save_fname, mode='w') as writer:
       writer.write_all(data)