from typing import Tuple, List, Generator

import torch

from word2vec.model.model import Word2vecModel, BATCH_SIZE, K_RATIO, CENTER, CONTEXT
from word2vec.model.word_distribution import adjusted_pruned_word_count, corpus, adjusted_unigram_distribution, \
    word_to_accumulated_probability

WINDOW_SIZE = 2  # Two words forwards and backwards from the center word

def tokenizer() -> Generator[Tuple[Tuple[str, str], Tuple[List[str], List[str]]], None, None]:
    """
    Generator, returns an encountered pair, and K_RATIO noise pairs
    TODO - how is this actually done in real life?
    """
    distribution_input = word_to_accumulated_probability()

    data = corpus()
    # Initializing the window
    window_length = 2 * WINDOW_SIZE + 1
    window = [next(data) for i in range(window_length)]
    center_word_index = WINDOW_SIZE  # Always
    try:
        while True:
            center_word = window[center_word_index]

            for i in range(window_length):
                if i == center_word_index:
                    continue

                encountered_pair = (center_word, window[i])
                noise_pairs = ([center_word] * K_RATIO, [adjusted_unigram_distribution(*distribution_input) for i in range(K_RATIO)])
                yield encountered_pair, noise_pairs

            # Update window:
            window.pop(0)
            window.append(next(data))

    except StopIteration as e:
        print("Done with epoch")


def train():
    words_to_index = {word: index for index, word in enumerate(adjusted_pruned_word_count().keys())}
    word_count = len(list(words_to_index.keys()))

    model = Word2vecModel(word_count=word_count, word_to_index_dict=words_to_index)
    my_tokenizer = tokenizer()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(1000):
        # Todo - loop the following, and wrap with stop iteration for the epoch:
        encountered_center_words: List[str] = []
        encountered_context_words: List[str] = []
        noise_center_words: List[str] = []
        noise_context_words: List[str] = []
        encountered_word_pairs = (encountered_center_words, encountered_context_words)
        noise_word_pairs = (noise_center_words, noise_context_words)
        for i in range(BATCH_SIZE):
            encountered_pair, curr_noise_word_pairs = next(my_tokenizer)
            encountered_center_words.append(encountered_pair[CENTER])
            encountered_context_words.append(encountered_pair[CONTEXT])
            noise_center_words += curr_noise_word_pairs[CENTER]
            noise_context_words += curr_noise_word_pairs[CONTEXT]

        loss = model(encountered_pairs=encountered_word_pairs, noise_pairs=noise_word_pairs)
        loss.backward()
        optimizer.step()
        print(loss.item())

    pass


if __name__ == '__main__':
    train()
