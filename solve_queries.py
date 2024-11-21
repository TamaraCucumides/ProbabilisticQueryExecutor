import pickle
import torch
import os

import utils


def get_relation(number, thresh=False):
    folder_path = "/home/tacucumides/storage/nbfnet-experiments/KnowledgeGraphCompletionV2/FB15k237-2/NBFNet/data"

    file_path = os.path.join(folder_path, f"{number}.pt")


    if os.path.exists(file_path):
        if thresh:
            matrix = torch.load(file_path)
            mask = (matrix < t)
            masked_matrix = matrix.masked_fill(mask, 0)
            return masked_matrix[:14504, :14504]
        return torch.load(file_path)[:14504, :14504]   

    else:
        print(f"Tensor file {number}.pt does not exist.")
        return None


def query_1p(query, top_k=False):
    """
    ok Receives a (e,r)
    """

    e = query[0]
    r = query[1]

    if type(r) is tuple:
        r = r[0]

    r_tensor = get_relation(r)

    if e == 99999 or e == -5:
        partial_result = r_tensor[0]
        for i in range(1, result_matrix.shape[0]):
            intermediate_result = partial_result + result_matrix[i] - (partial_result * result_matrix[i])
            answers = intermediate_result 
    else:
        answers = r_tensor[e]  

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_2p(query, top_k=False):
    """ ok Receives (e,r,r)"""
    e = query[0]

    if type(query[1]) is tuple:
        r1 = query[1][0]
        r2 = query[1][1]

    else:
        r1 = query[1]
        r2 = query[2]

    q1 = (e,r1)

    vec = query_1p(q1)

    r_2 = get_relation(r2)

    result_matrix = r_2 * vec.unsqueeze(1)

    partial_result = result_matrix[0]

    for i in range(1, result_matrix.shape[0]):
        intermediate_result = partial_result + result_matrix[i] - (partial_result * result_matrix[i])
        partial_result = intermediate_result

    if top_k:
        partial_result = mask_top_k(partial_result, k)

    return partial_result

def query_3p(query, top_k=False):
    """ ok Receives (e,r,r,r)"""
    e = query[0]

    if type(query[1]) is tuple:
        r1 = query[1][0]
        r2 = query[1][1]
        r3 = query[1][2]

    else:
        r1 = query[1]
        r2 = query[2]
        r3 = query[3]


    q1 = (e,r1,r2)

    vec = query_2p(q1)
    r_3 = get_relation(r3)

    result_matrix = r_3 * vec.unsqueeze(1)

    partial_result = result_matrix[0]
    for i in range(1, result_matrix.shape[0]):
        intermediate_result = partial_result + result_matrix[i] - (partial_result * result_matrix[i])
        partial_result = intermediate_result

    if top_k:
        partial_result = mask_top_k(partial_result, k)

    return partial_result

def query_2i(query, top_k=False):
    """ok Receives ((e1,r1), (e2,r2))"""

    v1 = query_1p(query[0])
    v2 = query_1p(query[1])

    answers = utils.intersection(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_3i(query, top_k=False):
    """ok Receives ((e1,r1),(e2,r2),(e3,r3))"""

    v1 = query_2i((query[0],query[1]))
    v2 = query_1p(query[2])

    answers = utils.intersection(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_ip(query, top_k=False):
    """ok Receives (((e1,r1),(e2,r2)),(r3))"""

    v1 = query_2i(query[0])
    r = query[1][0]

    r1 = get_relation(r)

    result_matrix = r1 * v1.unsqueeze(1)
    
    partial_result = result_matrix[0]
    for i in range(1, result_matrix.shape[0]):
        intermediate_result = partial_result + result_matrix[i] - (partial_result * result_matrix[i])
        partial_result = intermediate_result

    if top_k:
        partial_result = mask_top_k(partial_result, k)

    return partial_result

def query_pi(query, top_k=False):
    """ ok Receives (("e", ("r", "r")), ("e", ("r",)))"""

    q0 = (query[0][0], query[0][1][0], query[0][1][1])
    q1 = (query[1][0], query[1][1][0])

    v1 = query_2p(q0)
    v2 = query_1p(q1)

    answers = utils.intersection(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_2in(query, top_k=False):
    """ ok (("e", ("r",)), ("e", ("r", "n"))) """

    q1 = (query[0][0], query[0][1][0])
    v1 = query_1p(q1)

    q2 = (query[1][0], query[1][1][0])
    v2 = query_1p(q2)
    v2 = utils.negation(v2)

    answers = utils.intersection(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_3in(query, top_k=False):
    """ok (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n")))"""

    q1 = (query[0][0], query[0][1][0])
    q2 = (query[1], query[2])

    v1 = query_1p(q1)
    v2 = query_2in(q2)

    answers = utils.intersection(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_inp(query, top_k=False):
    """ ((("e", ("r",)), ("e", ("r", "n"))), ("r",)) """

    q1 = query[0]
    v1 = query_2in(q1)

    r1 = query[1][0]
    r_1 = get_relation(r1)

    result_matrix = r_1 * v1.unsqueeze(1)
    
    partial_result = result_matrix[0]
    for i in range(1, result_matrix.shape[0]):
        intermediate_result = partial_result + result_matrix[i] - (partial_result * result_matrix[i])
        partial_result = intermediate_result

    if top_k:
        partial_result = mask_top_k(partial_result, k)

    return partial_result

def query_pin(query, top_k=False):
    """ (("e", ("r", "r")), ("e", ("r", "n"))) """

    q1 = query[0]
    q2 = (query[1][0], query[1][1][0])

    v1 = query_2p(q1)

    v2 = query_1p(q2)
    v2 = utils.negation(v2)

    answers = utils.intersection(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_pni(query, top_k=False):
    """ (("e", ("r", "r", "n")), ("e", ("r",))) """

    q1 = (query[0][0], query[0][1][0], query[0][1][1])
    q2 = query[1]

    v1 = utils.negation(query_2p(q1))
    v2 = query_1p(q2)
    
    answers = utils.intersection(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_2u(query, top_k=False):
    """ (("e", ("r",)), ("e", ("r",)), ("u",)) """

    q1 = query[0]
    q2 = query[1]

    v1 = query_1p(q1)
    v2 = query_1p(q2)

    answers = utils.union(v1,v2)

    if top_k:
        answers = mask_top_k(answers, k)

    return answers

def query_up(query, top_k=False):
    """ ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)) """

    q1 = query[0]
    v1 = query_2u(q1)

    r1 = query[1]
    r_1 = get_relation(r1)

    result_matrix = r_1 * v1.unsqueeze(1)
    
    partial_result = result_matrix[0]
    for i in range(1, result_matrix.shape[0]):
        intermediate_result = partial_result + result_matrix[i] - (partial_result * result_matrix[i])
        partial_result = intermediate_result

    if top_k:
        partial_result = mask_top_k(partial_result, k)

    return partial_result

query2func = {"1p": query_1p,
"2p": query_2p,
"3p": query_3p,
"2i": query_2i,
"3i": query_3i, 
"ip": query_ip, 
"pi": query_pi, 
"2in": query_2in,
"3in": query_3in, 
"inp": query_inp,
"pin": query_pin, 
"pni": query_pni,
"2u": query_2u, 
"up": query_up}

def solve_query(tipo, queries, top_k=False):
    method = query2func[tipo]
    preds = {}

    for q in queries:
        pred = method(q)
        preds[q] = pred

    return preds


if __name__ == "__main__":
    test_3p = ((28, (4,)), (32, (2,)), (5, (3, "n")))

    ans = query_3in(test_3p)

    print(ans)