import pickle
import torch
import os

import utils
import solve_queries

num_entities = 14501

def solve_triangle_unravel(query, depth):
    solutions = dict()

    rel1 = solve_queries.get_relation(None)
    rel2 = solve_queries.get_relation(None)
    rel3 = solve_queries.get_relation(None)

    clock = []
    inv_clock = []

    for d in range(depth):
        pass

def solve_square_unravel(query, depth):
    solutions = dict()

    rel1 = solve_queries.get_relation(None)
    rel2 = solve_queries.get_relation(None)
    rel3 = solve_queries.get_relation(None)
    rel3 = solve_queries.get_relation(None)

    clock = []
    inv_clock = []
    pass

def solve_triangle_diss(query):
    """ Receives rel1,rel2,rel3, query is rel1(x,y)rel2(y,z)rel3(x,z)
    and target node is x """

    solutions = dict()

    rel1 = solve_queries.get_relation(None)
    rel2 = solve_queries.get_relation(None)
    rel3 = solve_queries.get_relation(None)

    # Plan 1 (join S and T, then join R)

    rel2_rel3 = None
    for x in range(num_entities):
        for y in range(num_entities):
            xy = 0
            for z in range(num_entities):
                xyz = rel2[y][z] * rel3[x][z]
                # update xy
            rel2_rel3[x][y] = xy
    
    result_plan1 = None
    for x in range(num_entities):
        x = 0
        for y in range(num_entities):
            xy = rel1[x][y] * rel2_rel3[x][y]
            # update x
        result_plan1[x] = x

    solutions["plan 1"] = result_plan1

    # Plan 2 (remove y)

    rel1_rel2 = None
    for x in range(num_entities):
        for z in range(num_entities):
            xz = 0
            for y in range(num_entities):
                xyz = rel1[x][y] * rel2[y][z]
                # update xz
            rel1_rel2[x][z] = xz
    
    result_plan1 = None
    for x in range(num_entities):
        x = 0
        for z in range(num_entities):
            xy = rel1[x][z] * rel1_rel2[x][z]
            # update x
        result_plan2[x] = x

    solutions["plan 2"] = result_plan2

    return solutions