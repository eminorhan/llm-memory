import time
import requests
import jsonlines
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

URL = ['https://astralcodexten.substack.com/p/a-columbian-exchange',
       'https://astralcodexten.substack.com/p/how-trustworthy-are-supplements',
       'https://astralcodexten.substack.com/p/chai-assistance-games-and-fully-updated',
       'https://astralcodexten.substack.com/p/universe-hopping-through-substack',
       'https://astralcodexten.substack.com/p/from-nostradamus-to-fukuyama',
       'https://astralcodexten.substack.com/p/why-is-the-central-valley-so-bad',
       'https://astralcodexten.substack.com/p/janus-gpt-wrangling',
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
       'https://fakenous.substack.com/p/does-history-cause-racism',
       'https://fakenous.substack.com/p/is-critical-thinking-epistemically',
       'https://fakenous.substack.com/p/is-this-racism',
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
       'https://betonit.substack.com/p/aaronson-on-feminism-my-reply',
       'https://betonit.substack.com/p/open-borders-is-a-fine-slogan',
       'https://betonit.substack.com/p/the-definition-of-feminism',
       'https://betonit.substack.com/p/rothbard-contra-the-demagogue',
       'https://betonit.substack.com/p/fear-of-feminism',
       'https://betonit.substack.com/p/safety-in-numbers-why-student-loan',
       'https://betonit.substack.com/p/reflections-on-the-caplan-singer',
       'https://betonit.substack.com/p/tyler-on-feminism-my-reply',
       'https://betonit.substack.com/p/dont-be-a-feminist-the-origin-story',
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
       'https://betonit.substack.com/p/reflections-on-italian-migration',
       'https://rychappell.substack.com/p/when-metaethics-matters',
       'https://rychappell.substack.com/p/a-multiplicative-model-of-value-pluralism',
       'https://rychappell.substack.com/p/pick-some-low-hanging-fruit',
       'https://rychappell.substack.com/p/puzzles-for-everyone',
       'https://rychappell.substack.com/p/how-useful-is-utilitarianism',
       'https://rychappell.substack.com/p/the-nietzschean-challenge-to-effective',
       'https://rychappell.substack.com/p/ethically-alien-thought-experiments',
       'https://rychappell.substack.com/p/utilitarianism-and-abortion',
       'https://rychappell.substack.com/p/review-of-what-we-owe-the-future',
       'https://rychappell.substack.com/p/billionaire-philanthropy',
       'https://rychappell.substack.com/p/double-or-nothing-existence-gambles',
       'https://rychappell.substack.com/p/constraints-and-candy',
       'https://rychappell.substack.com/p/ethics-as-solutions-vs-constraints',
       'https://rychappell.substack.com/p/is-non-consequentialism-self-effacing',
       'https://rychappell.substack.com/p/the-fine-tuning-god-problem',
       'https://rychappell.substack.com/p/meat-externalities',
       'https://rychappell.substack.com/p/caplans-conscience-objection-to-utilitarianism',
       'https://rychappell.substack.com/p/agency-and-epistemic-cheems-mindset',
       'https://rychappell.substack.com/p/buddhism-and-utilitarianism',
       'https://rychappell.substack.com/p/consequentialism-beyond-action',
       'https://rychappell.substack.com/p/emergency-ethics',
       'https://rychappell.substack.com/p/the-strange-shortage-of-moral-optimizers',
       'https://rychappell.substack.com/p/moral-truth-without-substance',
       'https://rychappell.substack.com/p/deontic-pluralism',
       'https://rychappell.substack.com/p/the-birth-of-population-ethics',
       'https://rychappell.substack.com/p/do-you-really-exist-over-time',
       'https://rychappell.substack.com/p/parfits-triple-theory',
       'https://rychappell.substack.com/p/level-up-impartiality',
       'https://rychappell.substack.com/p/rational-irrationality-and-blameless',
       'https://rychappell.substack.com/p/utilitarianism-and-the-personal-perspective',
       'https://rychappell.substack.com/p/priority-and-aggregation',
       'https://rychappell.substack.com/p/how-utilitarians-value-individuals',
       'https://rychappell.substack.com/p/against-egoism-and-subjectivism',
       'https://rychappell.substack.com/p/theory-driven-applied-ethics',
       'https://rychappell.substack.com/p/beneficentrism',
       'https://rychappell.substack.com/p/beware-status-quo-risks',
       'https://rychappell.substack.com/p/utilitarianism-and-reflective-equilibrium',
       'https://rychappell.substack.com/p/utilitarianism-debate-with-michael'
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
              # remove short texts (pars)
              if len(line.text) > 100:
                     sentences = sent_tokenize(line.text)
                     for sent in sentences:
                            # remove short sentences
                            if len(sent) > 100:
                                   data.append({"sent": sent, "sent_id": par_id, "url_id": url_id})
                                   par_id += 1
       url_id += 1

# write to jsonl file
with jsonlines.open('data.jsonl', mode='w') as writer:
       writer.write_all(data)