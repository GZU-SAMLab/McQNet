# this is taken from here: https://github.com/oscarknagg/few-shot/blob/2830f29e757a0db35f998112f5c866ab365f6ef2/few_shot/core.py
import torch
import numpy as np
from torch.utils.data import Sampler


def make_label_idx2indicies(dataset):
    label_idx2indicies = {}
    for i in range(len(dataset)):
        label_idx = dataset.get_label_idx(i)
        if label_idx in label_idx2indicies:
            label_idx2indicies[label_idx].append(i)
        else:
            label_idx2indicies[label_idx] = [i]
    return label_idx2indicies


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int,
                 n_shot: int,
                 n_way: int,
                 n_query: int,
                 ):
        """        
        PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.
        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.
        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.
        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            n_way: int. Number of classes in the n-shot classification tasks.
            n_query: int. Number query samples for each class in the n-shot classification tasks.
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        self.label_idx2indicies = make_label_idx2indicies(dataset)

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            # choose classes to use in this episode
            episode_classes = np.random.choice(
                list(self.label_idx2indicies.keys()),
                self.n_way, replace=False)

            # choose indicies to use in this example
            # select support + query
            support_indicies = []
            query_indicies = []
            for cls_idx in episode_classes:
                try:
                    sample_indicies = np.random.choice(
                        self.label_idx2indicies[cls_idx],
                        self.n_shot + self.n_query, replace=False)
                except:
                    import warnings
                    warnings.warn("Ops! proably not enough samples, so had to allow duplicates for this one.")
                    sample_indicies = np.random.choice(
                        self.label_idx2indicies[cls_idx],
                        self.n_shot + self.n_query, replace=True)

                support_indicies += sample_indicies[0:self.n_shot].tolist()
                query_indicies += sample_indicies[self.n_shot:self.n_shot + self.n_query].tolist()

            batch = support_indicies + query_indicies
            yield np.stack(batch)
