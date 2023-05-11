import math
import random
from collections import Counter, defaultdict
from time import time
from typing import Iterable, Dict, Tuple

import joblib
import torch
import torch.nn.functional as F
import re
import numpy as np
from sortedcontainers import SortedDict

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

RARE_WORD_THRESHOLD = 100

memory = joblib.Memory("cachedir")

t0 = 0


def corpus(rare_words: set = None):
    """
    A generator for the words in the corpus.
    rare words - a set of words to filter out if they appear.
    """
    from datasets import load_dataset

    global t0
    corpus_dataset = load_dataset("wikipedia", "20220301.en", split='train')
    t0 = time()
    print(t0)
    for article in corpus_dataset:
        # Removing all the non-letter characters in the article's text
        if int(article['id']) % 1000 == 0:
            print(f"{article['id']}")

        text = re.sub(r'[^a-zA-Z ]', ' ', article['text']).lower()
        for word in text.split():
            if not rare_words or word not in rare_words:
                yield word
            # TODO - Should I yield a rare word token otherwise? Do I mind the next non-rare words being interpreted as closer?


def zero():
    return 0

@memory.cache
def word_count():
    """
    Dict mapping word to its count in the corpus.
    """
    # Using zero instead of a simple lambda because pickle which memory.cache uses fails on lambdas.
    word_count_dict = defaultdict(zero)

    g = corpus()

    try:
        next_word = next(g)
        counter = 0

        while True:
            for i in range(1000000):
                # Inner loop to not spam too much with prints
                word_count_dict[next_word] += 1

                next_word = next(g)
            counter += 1
            print(f"Counter: {counter}, len: {len(word_count_dict)}")
    except:
        print("Excepted :)")
        pass

    return word_count_dict


@memory.cache
def pruned_word_count() -> Tuple[Dict[str, int], set]:
    """
    Dict mapping word to its count in the corpus, excluding rare words. Produces Also a set of rare words.
    """
    word_count_dict = word_count()

    word_count_without_rare_words = dict()
    rare_words = set()

    for word in word_count_dict:
        if word_count_dict[word] > RARE_WORD_THRESHOLD:
            word_count_without_rare_words[word] = word_count_dict[word]
        else:
            rare_words.add(word)

    return word_count_without_rare_words, rare_words


@memory.cache
def adjusted_pruned_word_count() -> Dict[str, float]:
    """
    The word count adjusted to the power of 3/4.
    """
    word_count_without_rare_words, _ = pruned_word_count()

    adjusted_word_count_without_rare_words = dict()

    for word in word_count_without_rare_words:
        adjusted_word_count_without_rare_words[word] = math.pow(word_count_without_rare_words[word], 0.75)
        # TODO - should I cap the probability here by some constant?

    return adjusted_word_count_without_rare_words


@memory.cache()
def word_to_accumulated_probability() -> Tuple[SortedDict[float, str], float]:
    """
    Returns a dictionary mapping a word to it's accumulated probability distribution from 0 to the given output of
    normalizing sum.
    Relies on the items in a dictionary being kept in the insertion order, which is true since python 3.6.
    """
    adjusted_count = adjusted_pruned_word_count()

    word_to_accumulated_probability_dict = dict()

    accumulated_probability = 0
    for word, value in sorted(adjusted_count.items(), key=lambda x: x[1], reverse=True):
        word_to_accumulated_probability_dict[accumulated_probability] = word
        accumulated_probability += value

    normalizing_sum = sum(adjusted_count.values())

    print(f"normalizing_sum: {normalizing_sum}. accumulated_probability: {accumulated_probability} ")

    return SortedDict(word_to_accumulated_probability_dict), accumulated_probability


def adjusted_unigram_distribution(word_to_accumulated_probability_dict, normalizing_sum):
    """
    Produces words according to unigram distribution shifted by the power of 3/4 and normalized.
    """
    rand = random.uniform(0, normalizing_sum)
    index = word_to_accumulated_probability_dict.bisect(rand)
    interesting_key = word_to_accumulated_probability_dict.iloc[index - 1]
    #print(f"Index: {index}, {rand}, value before: {word_to_accumulated_probability_dict.iloc[index - 1]} value after: {word_to_accumulated_probability_dict.iloc[index]}")
    return word_to_accumulated_probability_dict[interesting_key]


print(adjusted_unigram_distribution(*word_to_accumulated_probability()))