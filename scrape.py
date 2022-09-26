import time
import requests
import jsonlines
from bs4 import BeautifulSoup

URL = [
       'https://astralcodexten.substack.com/p/unpredictable-reward-predictable',
       'https://astralcodexten.substack.com/p/i-won-my-three-year-ai-progress-bet',
       'https://astralcodexten.substack.com/p/links-for-september-2022',
       'https://astralcodexten.substack.com/p/book-review-contest-2022-winners',
       'https://astralcodexten.substack.com/p/the-prophet-and-caesars-wife',
       'https://astralcodexten.substack.com/p/billionaires-surplus-and-replaceability',
       'https://astralcodexten.substack.com/p/your-book-review-kora-in-hell',
       'https://astralcodexten.substack.com/p/effective-altruism-as-a-tower-of',
       'https://astralcodexten.substack.com/p/book-review-what-we-owe-the-future',
       'https://astralcodexten.substack.com/p/your-book-review-1587-a-year-of-no',
       'https://astralcodexten.substack.com/p/highlights-from-the-comments-on-subcultures',
       'https://astralcodexten.substack.com/p/skills-plateau-because-of-decay-and',
       'https://astralcodexten.substack.com/p/mantic-monday-81522',
       'https://astralcodexten.substack.com/p/your-book-review-god-emperor-of-dune',
       'https://astralcodexten.substack.com/p/will-nonbelievers-really-believe',
       'https://astralcodexten.substack.com/p/a-cyclic-theory-of-subcultures',
       'https://astralcodexten.substack.com/p/why-not-slow-ai-progress',
       'https://astralcodexten.substack.com/p/your-book-review-exhaustion',
       'https://astralcodexten.substack.com/p/absurdity-bias-neom-edition',
       'https://astralcodexten.substack.com/p/slightly-against-underpopulation',
       'https://astralcodexten.substack.com/p/model-city-monday-8122',
       'https://astralcodexten.substack.com/p/your-book-review-viral',
       'https://astralcodexten.substack.com/p/forer-statements-as-updates-and-affirmations',
       'https://astralcodexten.substack.com/p/elk-and-the-problem-of-truthful-ai',
       'https://astralcodexten.substack.com/p/your-book-review-the-society-of-the',
       'https://fakenous.substack.com/p/americas-unjust-drug-war',
       'https://fakenous.substack.com/p/immoral-rulers',
       'https://fakenous.substack.com/p/sense-data',
       'https://fakenous.substack.com/p/stupendously-awful-ideas-621-cancelling',
       'https://fakenous.substack.com/p/is-there-a-right-to-own-a-gun',
       'https://fakenous.substack.com/p/your-opponents-dont-agree-with-you',
       'https://fakenous.substack.com/p/non-egalitarianism',
       'https://fakenous.substack.com/p/whats-a-fair-price-or-youre-not-underpaid',
       'https://fakenous.substack.com/p/causation-as-simultaneous-and-continuous',
       'https://fakenous.substack.com/p/political-activism-whats-the-point',
       'https://fakenous.substack.com/p/arbitrary-foundations',
       'https://fakenous.substack.com/p/the-allure-of-social-predators',
       'https://fakenous.substack.com/p/the-principle-of-inferential-justification',
       'https://fakenous.substack.com/p/who-cares-about-diversity',
       'https://fakenous.substack.com/p/the-problem-of-defeasible-justification',
       'https://fakenous.substack.com/p/time-travel-is-ridiculously-impossible',
       'https://fakenous.substack.com/p/naturalism-the-problem-of-moral-knowledge',
       'https://fakenous.substack.com/p/crappy-thesis-movies',
       'https://fakenous.substack.com/p/freedom-determinism-are-incompatible',
       'https://fakenous.substack.com/p/roe-v-wade-who-was-right',
       'https://fakenous.substack.com/p/neglected-cool-future-technologies',
       'https://fakenous.substack.com/p/direct-realism-and-the-brain-in-a-vat-argument',
       'https://fakenous.substack.com/p/why-we-cancel',
       'https://fakenous.substack.com/p/the-problem-of-memory-knowledge',
       'https://fakenous.substack.com/p/why-we-love-evil-ideas',
       'https://betonit.substack.com/p/the-caplan-singer-debate-my-opening',
       'https://betonit.substack.com/p/dont-be-a-feminist-highlights',
       'https://betonit.substack.com/p/judge-this-book-by-its-cover',
       'https://betonit.substack.com/p/dont-be-a-feminist-releases-today',
       'https://betonit.substack.com/p/how-immigrants-became-democrats',
       'https://betonit.substack.com/p/fighting-inflation',
       'https://betonit.substack.com/p/safe-sounds-good',
       'https://betonit.substack.com/p/i-want-to-hold-your-hand',
       'https://betonit.substack.com/p/reflections-on-katz-witch-trial',
       'https://betonit.substack.com/p/the-soviet-dictionary-on-socialism',
       'https://betonit.substack.com/p/blame-the-principals',
       'https://betonit.substack.com/p/republicans-finally-turn-on-the-minimum',
       'https://betonit.substack.com/p/no-deal-how-politics-really-works',
       'https://betonit.substack.com/p/cancelling-student-debt-is-unforgivable',
       'https://betonit.substack.com/p/the-final-freedom',
       'https://betonit.substack.com/p/raiding-woke-capital',
       'https://betonit.substack.com/p/the-tautology-of-chemical-imbalance',
       'https://betonit.substack.com/p/the-distributive-distraction',
       'https://betonit.substack.com/p/school-choice-sorry-i-underrated',
       'https://betonit.substack.com/p/incidence-not-insanity',
       'https://betonit.substack.com/p/safety-in-numbers-why-student-loan',
       'https://betonit.substack.com/p/the-united-states-party-central',
       'https://betonit.substack.com/p/inflation-for-merit',
       'https://betonit.substack.com/p/starting-iunbeatablei',
       'https://betonit.substack.com/p/reflections-on-italian-migration'
       ]

data = []
par_id = 0
url_id = 0

for url in URL:
       time.sleep(1)  # to prevent too many requests error
       print(url)
       r = requests.get(url)
       soup = BeautifulSoup(r.content, 'html.parser')  # parse html
       s = soup.find('div', class_='available-content')
       lines = s.find_all('p')
       for line in lines:
              # remove short texts
              if len(line.text) > 50:
                     data.append({"par": line.text, "par_id": par_id, "url_id": url_id})
                     par_id += 1
       url_id += 1

# write to jsonl file
with jsonlines.open('data.jsonl', mode='w') as writer:
       writer.write_all(data)