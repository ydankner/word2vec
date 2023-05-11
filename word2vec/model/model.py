import torch
from torch.nn import Sigmoid
from torch import rand, ones, Tensor
from typing import List, Tuple, Dict

from base import BaseModel


WORD_SPACE_DIM = 300
BATCH_SIZE = 10  # How many word pairs will be taken to account in each forward pass?
K_RATIO = 10  # How many noise pairs for each real pair.

CENTER = 0
CONTEXT = 1

class Word2vecModel(BaseModel):
    def __init__(self, word_count: int, word_to_index_dict: Dict[str, int]):
        super().__init__()
        word_vector_shape = (
            2,  # For each word, we want a context vector and a center vector
            word_count,
            WORD_SPACE_DIM
        )

        # Initializing a random tensor for each word.
        word_vectors_unshifted = rand(word_vector_shape, requires_grad=True)

        # Since rand returns a uniform distribution between 0-1, we need to shift it by -0.5 to fit to the initial
        # randomized word vectors:
        self.word_vectors = torch.nn.Parameter(word_vectors_unshifted - (ones(word_vector_shape) / 2))
        self.word_to_index_dict = word_to_index_dict

    def select_indices_from_words(self, words: List[str], word_vectors: Tensor) -> Tensor:
        """
        Given a list of words, selects these indices from the first dimension of the given tensor
        """
        word_indices = torch.tensor([self.word_to_index_dict[word] for word in words])
        relevant_word_vectors = torch.index_select(word_vectors, 0, word_indices)
        return relevant_word_vectors

    def products_of_center_and_context(self, word_pairs: Tuple[List[str], List[str]]) -> Tensor:
        """
        Given two lists of context and center words with the same length, return a one-dimensional Tensor of the same
        length, computing pairwise the dot multiplication of the appropriate word vectors for each index of the given
        words.
        """
        center_word_vectors = self.word_vectors[CENTER]
        context_word_vectors = self.word_vectors[CONTEXT]

        encountered_center_word_vectors = self.select_indices_from_words(word_pairs[CENTER], center_word_vectors)
        encountered_context_word_vectors = self.select_indices_from_words(word_pairs[CONTEXT], context_word_vectors)

        # Computing the vector of the dot products of every encountered center word with every encountered context word:
        return (encountered_context_word_vectors * encountered_center_word_vectors).sum(dim=1)

    def forward(self, encountered_pairs: Tuple[List[str], List[str]], noise_pairs: Tuple[List[str], List[str]]) -> Tensor:
        """
        The forward pass simply computes the scalar loss.

        The loss is computed as the negative of equation 4 from article 1310.4546 on arxiv (Distributed Representations
        of Words and Phrases and their Compositionality).
        """
        encountered_pairs_products = self.products_of_center_and_context(word_pairs=encountered_pairs)
        noise_pairs_products = self.products_of_center_and_context(word_pairs=noise_pairs)

        encountered_pairs_objective = torch.log(torch.sigmoid(encountered_pairs_products)).sum()
        # Notice the "-" to negate the noise pairs:
        noise_pairs_objective = torch.log(torch.sigmoid(-noise_pairs_products)).sum()

        objective = encountered_pairs_objective + noise_pairs_objective
        loss = -objective

        return loss
