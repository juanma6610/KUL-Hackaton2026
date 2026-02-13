Information about the dataset:

https://sharedtask.duolingo.com/ 

This shared task is a research challenge organized by Duolingo in collaboration with the WNGT workshop at ACL 2020. Its goal was to improve automatic translation systems so they can produce many correct translations for a given English sentence, similar to how Duolingo’s learners might respond, instead of just one output like typical machine translation systems.

Background:
Duolingo uses translation exercises in its language learning platform. Because there are many valid ways to translate a sentence (e.g., multiple correct answers from learners), the task focuses on generating high‑coverage sets of plausible translations in five languages: Portuguese, Hungarian, Japanese, Korean, and Vietnamese. These datasets include translations weighted by how frequently real learners choose them.

Given an English sentence, participants generate a set of possible translations in a target language.

A strong baseline machine translation reference is provided (from Amazon).

Translations are evaluated against human‑curated, weighted sets of acceptable translations.

Data came from Duolingo courses with English prompts and many accepted learner translations, with training/dev/test splits for each language pair:
1) English → Portuguese
2) English → Hungarian
3) English → Japanese
4) English → Korean
5) English → Vietnamese

1. File Naming Overview

The files follow a pattern:

	[split].[source]_[target].[date].[type].txt

Here: 
- split: train, dev (validate), test — the dataset split.
- source_target: e.g., en_hu = English → Hungarian.
- date: the release date of that dataset.
- type:
	- gold.txt → the human-validated (gold standard) translations
	- aws_baseline.pred.txt → predictions from the AWS baseline system

2. Dataset Structure

Each .txt file usually has tab-separated fields, typical for the Duolingo shared task. For gold.txt datasets, we will have the following format: 

	prompt_e7f806e856f45836d3a29df816ead23b|do you have animals?
	vannak állataid?|0.17335260969298194

Here: 
1) ID – prompt_e7f806e856f45836d3a29df816ead23b
	- Unique identifier for the sentence prompt.
	- Used to match predictions to the reference.

2) Source sentence (English) – do you have animals?
	- The prompt that learners are translating.

3) Target sentence (Translation) – vannak állataid?
	- One valid translation in the target language (here Hungarian).

4) Weight / Frequency – 0.17335260969298194
	- Represents how often learners produced this translation.
	- Values are normalized between 0 and 1 so that all translations for a prompt sum roughly to 1.
	- Used in evaluation to weight predictions: generating more frequent translations is “worth more points”.

Prediction files (*_aws_baseline.pred.txt) are simpler in format:

	prompt_e7f806e856f45836d3a29df816ead23b|van állatod?

Here: 
1) One predicted translation per prompt ID.
2) The evaluation checks how well this matches any gold translations, weighted by their frequency.

With this dataset, a lot of additional data science tasks can be done! Think out of the box, and impress us during the datathon pitches!!
