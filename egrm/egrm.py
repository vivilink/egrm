### import packages
import tskit
import tqdm
import math
import numpy as np
import pandas as pd

### import C extension
import matrix


# exports [matrix._matrix_C_API] object into a 2d numpy array mat_C: [matrix._matrix_C_API] object initiated by
# matrix.new_matrix, only stores the upper triangle elements of a square matrix N: the number of columns/rows
def mat_C_to_ndarray(mat_C, N):
    buffer = matrix.export_ndarray(mat_C)
    buffer = buffer + np.transpose(buffer) - np.diag(np.diag(buffer))
    matrix.destroy_matrix(mat_C)
    return buffer


### defines [Gmap] object which maps the physical position (in bp) into genetic position (in unit of 10^-6 cM)
### can be initiated by Gmap(filename), where filename is a (comma/space/tab separated) three-column file 
### with first column specifying the physical position in bp and the third column specifying the genetic position in cM. 
### The second column is not used. The first line will always be ignored as the header.
class Gmap:
    def __init__(self, filename):
        if filename is None:
            self.mapped = False
            return
        self.table = pd.read_csv(filename, sep=None, engine='python')
        self.pos = self.table.iloc[:, 0].astype(int)
        self.gpos = self.table.iloc[:, 2].astype(float) * 1e6
        self.max = self.table.shape[0]
        self.i = 0
        self.mapped = True

    def __call__(self, pos):
        if self.mapped == False:
            return pos
        while (self.i > 0 and pos < self.pos[self.i - 1]):
            self.i -= 1
        while (self.i < self.max and pos > self.pos[self.i]):
            self.i += 1
        if self.i == 0:
            return 0
        if self.i >= self.max:
            return self.gpos[self.max - 1]
        A = pos - self.pos[self.i - 1]
        B = (self.gpos[self.i] - self.gpos[self.i - 1]) / (self.pos[self.i] - self.pos[self.i - 1])
        C = self.gpos[self.i - 1]
        return A * B + C


### main functions

def varGRM_C(tree, num_samples):
    """
    Calculate normalized eGRM for one tree of ARG
    @param num_samples: number of haploid samples
    @param tree: tskit.trees.tree
    @return: np.array normalized eGRM for tree, float expected number of mutations on tree
    """
    egrm_one_tree, total_mu_one_tree = egrm_one_tree_no_normalization_C(tree=tree, num_samples=num_samples)

    egrm_one_tree /= total_mu_one_tree

    egrm_one_tree -= egrm_one_tree.mean(axis=0)
    egrm_one_tree -= egrm_one_tree.mean(axis=1, keepdims=True)

    return egrm_one_tree, total_mu_one_tree


def egrm_one_tree_no_normalization_C(tree,
                                     num_samples,
                                     gmap=Gmap(None),
                                     left=0,
                                     right=math.inf,
                                     g=(lambda x: 1 / (x * (1 - x)))):
    """
    Extracted from original varGRM_C function. Calculates an unnormalized eGRM (no division by mu(G) and
    no centering by column or row.
    @param tree: tskit.trees.tree One tree of ARG
    @param num_samples: int Number of haplotypes is sample
    @param gmap:
    @param left:
    @param right:
    @param g: function used to get something like expected number of descendants from a branch?
    @return: np.array, float unnormalized eGRM for one tree, expected number of mutations on that tree
    """
    num_samples = num_samples
    egrm_C = matrix.new_matrix(num_samples)

    total_mu_one_tree = 0
    tree_span = - gmap(max(left, tree.interval[0])) + gmap(min(right, tree.interval[1]))
    if tree_span <= 0:
        raise ValueError("l is negative (egrm)")
    if tree.total_branch_length == 0:
        raise ValueError("branch length is zero (egrm)")

    for c in tree.nodes():
        descendants = list(tree.samples(c))
        n = len(descendants)
        if n == 0 or n == num_samples:
            continue
        branch_len = max(0, tree.time(tree.parent(c)) - tree.time(c))
        mu = tree_span * branch_len * 1e-8
        p = float(n / num_samples)
        matrix.add_square(egrm_C, descendants, mu * g(p))
        total_mu_one_tree += mu

    egrm = mat_C_to_ndarray(egrm_C, num_samples)

    return egrm, total_mu_one_tree


# computes the mean TMRCA (mTMRCA) based on the tskit TreeSequence [trees].
# trees: tskit TreeSequence object.
# log: tqdm log file path
# left, right: leftmost and rightmost positions (in unit of base pair) between which the mTMRCA is computed.
# gmap: Gmap object that maps the physical position (in unit of base pair) into genetic position (in unit of 10^-6 centimorgan).
# sft: True/False variable indicating whether the first tree is to be skipped. Not recommended to use together with [left] and [right] options.
def mTMRCA_C(trees, log=None,
             left=0, right=math.inf,
             gmap=Gmap(None), sft=False):
    N = trees.num_samples
    mtmrca_C = matrix.new_matrix(N)
    tmp = 0
    total_l = 0
    pbar = tqdm.tqdm(total=trees.num_trees,
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                     miniters=trees.num_trees // 100,
                     file=log)

    trees_ = trees.trees()
    if sft: next(trees_)

    for tree in trees_:
        pbar.update(1)
        l = - gmap(max(left, tree.interval[0])) + gmap(min(right, tree.interval[1]))
        if l <= 0: continue
        if tree.total_branch_length == 0: continue

        height = 0
        for c in tree.nodes():
            descendants = list(tree.samples(c))
            n = len(descendants)
            if (n == 0 or n == N): continue
            t = tree.time(tree.parent(c)) - tree.time(c)
            height = max(height, tree.time(tree.parent(c)))
            matrix.add_square(mtmrca_C, descendants, t * l)
        tmp += height * l
        total_l += l

    mtmrca = mat_C_to_ndarray(mtmrca_C, N)
    mtmrca = tmp - mtmrca
    mtmrca /= total_l
    pbar.close()
    return mtmrca, total_l


### without C extension

def egrm_one_tree_no_normalization(tree,
                                   num_samples,
                                   gmap=Gmap(None),
                                   left=0,
                                   right=math.inf,
                                   g=(lambda x: 1 / (x * (1 - x)))):
    """
    Extracted from original non-C version of varGRM function. Calculates an unnormalized eGRM (no division by mu(G) and
    no centering by column or row.

    @param tree: tskit.trees.tree One tree of ARG
    @param num_samples: int Number of haplotypes is sample
    @param gmap:
    @param left:
    @param right:
    @param g: function used to get something like expected number of descendants from a branch?
    @return: np.array, float unnormalized eGRM for one tree, expected number of mutations on that tree
    """
    tree_span = - gmap(max(left, tree.interval[0])) + gmap(min(right, tree.interval[1]))
    if tree_span <= 0:
        raise ValueError("l is negative (egrm)")
    if tree.total_branch_length == 0:
        raise ValueError("branch length is zero (egrm)")

    total_mu_one_tree = 0
    cov = np.zeros([num_samples, num_samples])

    for c in tree.nodes():
        descendants = list(tree.samples(c))
        n = len(descendants)
        if n == 0 or n == num_samples:
            continue

        # I removed alim and rlim from following line, could cause problem
        branch_len = max(0, tree.time(tree.parent(c)) - tree.time(c))

        mu = tree_span * branch_len * 1e-8

        p = float(n / num_samples)
        cov[np.ix_(descendants, descendants)] += mu * g(p)
        total_mu_one_tree += mu

    return cov, total_mu_one_tree


# the non-C version of varGRM_C
def varGRM(tree, num_samples):
    """
    Calculate normalized eGRM for one tree of ARG

    @param num_samples: number of haploid samples
    @param tree: tskit.trees.tree
    @return: np.array normalized eGRM for tree, float expected number of mutations on tree
    """

    egrm_one_tree, total_mu_one_tree = egrm_one_tree_no_normalization(tree=tree, num_samples=num_samples)

    # normalize
    egrm_one_tree /= total_mu_one_tree
    egrm_one_tree -= egrm_one_tree.mean(axis=0)
    egrm_one_tree -= egrm_one_tree.mean(axis=1, keepdims=True)

    return egrm_one_tree, total_mu_one_tree


# the non-C version of mTMRCA_C
def mTMRCA(trees, log=None,
           left=0, right=math.inf,
           gmap=Gmap(None), sft=False):
    N = trees.num_samples
    mtmrca = np.zeros([N, N])
    tmp = 0
    total_l = 0
    pbar = tqdm.tqdm(total=trees.num_trees,
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                     miniters=trees.num_trees // 100,
                     file=log)

    trees_ = trees.trees()
    if sft: next(trees_)

    for tree in trees_:
        pbar.update(1)
        l = - gmap(max(left, tree.interval[0])) + gmap(min(right, tree.interval[1]))
        if l <= 0: continue
        if tree.total_branch_length == 0: continue

        height = 0
        for c in tree.nodes():
            descendants = list(tree.samples(c))
            n = len(descendants)
            if (n == 0 or n == N): continue
            t = tree.time(tree.parent(c)) - tree.time(c)
            height = max(height, tree.time(tree.parent(c)))
            mtmrca[np.ix_(descendants, descendants)] += t * l
        tmp += height * l
        total_l += l

    mtmrca = tmp - mtmrca
    mtmrca /= total_l
    pbar.close()
    return mtmrca, total_l
