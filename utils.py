import torch

struct2type = {
        ("e", ("r",)): "1p",
        ("e", ("r", "r")): "2p",
        ("e", ("r", "r", "r")): "3p",
        (("e", ("r",)), ("e", ("r",))): "2i",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
        ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
        ((("e", ("r",)), ("e", ("r","r"))), ("r",)): "i2p",
        (("e", ("r", "r")), ("e", ("r",))): "pi",
        (("e", ("r",)), ("e", ("r", "n"))): "2in",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
        ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
        (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
        (("e", ("r", "r", "n")), ("e", ("r",))): "pni",
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2u-DNF",
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up-DNF",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",)): "2u-DM",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r")): "up-DM",
        (("e", ("r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r"))): "5pi",
        (('e', ('r', 'r')), ('e', ('r', 'r'))): "2pi",
        (('e', ('r', 'r', 'r')), ('e', ('r', 'r', 'r'))): "3pi",
        (('e', ('r', 'r', 'r', 'r')), ('e', ('r', 'r', 'r', 'r'))): "4pi",
        (('e', ('r', 'r', 'r', 'r', 'r', 'r')), ('e', ('r', 'r', 'r', 'r', 'r', 'r'))): "6pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r"))): "7pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r"))): "8pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r"))): "9pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))): "10pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))): "11pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))): "12pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))): "13pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))): "14pi",
        (("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r")), ("e", ("r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))): "15pi",
        (((('e', ('r',)), ('e', ('r',))), ('r',)), ((('e', ('r',)), ('e', ('r',))), ('r',))): "2ip",
        (((('e', ('r',)), ('e', ('r', 'r'))), ('r',)), ((('e', ('r',)), ('e', ('r', 'r'))), ('r',))): "3ip"
    }

def intersection(vec_1, vec_2, tnorm="godel"):
    #return torch.min(vec_1, vec_2)
    return vec_1 * vec_2

def negation(vec_1, tnorm="godel"):
    return torch.abs(1 - vec_1)

def union(vec_1, vec_2, tnorm="godel"):
    #return torch.max(vec_1, vec_2)
    return 1-(1-vec_1)*(1-vec_2)

def disjunction_over_all(vec, matrix, tnorm="godel"):
    # Disjunction over all matrix
    return None

def mask_answers(ents, mask_size):
    """ Receives a list of ents [1,5] 
    and returns a mask of size mask_size"""

    mask = torch.zeros(mask_size, dtype=torch.bool)
    for num in ents:
        if num < mask_size:
            mask[num] = True
    return mask

def mask_top_k(answers, k):
    _, ranking = torch.sort(answers, descending=True)
    top_k_mask = torch.zeros_like(answers, dtype=torch.bool)
    top_k_mask[ranking[:k]] = True

    answers[~top_k_mask] = 0

    return answers 

def tensor_intersection(indices1, indices2):
    intersection_mask = torch.isin(indices1, indices2)
    intersection = indices1[intersection_mask]

    return intersection.tolist()

def filter_row(row, thresh):
    mask = row > thresh
    indices = torch.nonzero(mask).squeeze()

    return indices

def sparse_matrix(relation_tensor, thresh):
    sparse_dict = dict()

    for x in relation_tensor.shape[0]:
        sparse_dict[x] = filter_row(relation_tensor[x], thresh)

    return sparse_dict
