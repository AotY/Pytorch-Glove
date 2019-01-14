import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_normal

"""
2014:
GloVe: Global Vectors for Word Representation
"""

class GloVe(nn.Module):
    def __init__(self, co_occur, embed_size, x_max=100, alpha=0.75):
        """
        :param co_occur: Co-occurrence ndarray with shape of [num_classes, num_classes]
        :param embed_size: embedding size
        :param x_max: An int representing cutoff of the weighting function
        :param alpha: Ant float parameter of the weighting function
        """

        super(GloVe, self).__init__()

        self.embed_size = embed_size

        self.x_max = x_max
        self.alpha = alpha

        ''' co_occur Matrix is shifted in order to prevent having log(0) '''
        self.co_occur = co_occur + 1.0

        [self.num_classes, _] = self.co_occur.shape

        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.in_embed.weight = xavier_normal(self.in_embed.weight)

        self.in_bias = nn.Embedding(self.num_classes, 1)
        self.in_bias.weight = xavier_normal(self.in_bias.weight)

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.out_embed.weight = xavier_normal(self.out_embed.weight)

        self.out_bias = nn.Embedding(self.num_classes, 1)
        self.out_bias.weight = xavier_normal(self.out_bias.weight)

    def forward(self, input, output):
        """
        :param input: An array with shape of [batch_size] of int type
        :param output: An array with shape of [batch_size] of int type
        :return: loss estimation for Global Vectors word representations
                 defined in nlp.stanford.edu/pubs/glove.pdf
        """

        batch_size = len(input)

        co_occurences = np.array([self.co_occur[input[i], output[i]] for i in range(batch_size)])
        co_occurences = torch.from_numpy(co_occurences).float()

        # a weighting function f(X_ij)
        weights = np.array([self._weight(var) for var in co_occurences])
        weights = torch.from_numpy(weights).float()

        input = torch.from_numpy(input)
        input_embed = self.in_embed(input)
        input_bias = self.in_bias(input)

        output = torch.from_numpy(output)
        output_embed = self.out_embed(output)
        output_bias = self.out_bias(output)

        # weighted least squares regression model.
        return (torch.pow(
            ((input_embed * output_embed).sum(1) + input_bias + output_bias).squeeze(1) - \
            torch.log(co_occurences), 2
        ) * weights).sum()

    def _weight(self, x):
        return 1 if x > self.x_max else (x / self.x_max) ** self.alpha

    def embeddings(self):
        return self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
