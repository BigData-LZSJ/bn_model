#!/bin/python
from collections import deque

class Vertex_E(object):
    def __init__(self, idx, prop, creditscore, rating):
        self.idx = idx
        self.prop = prop
        self.creditscore = creditscore
        self.rating = rating

class Vertex_P(object):
    def __init__(self, idx, prop, cerno):
        self.idx = idx
        self.prop = prop
        self.cerno = cerno

class Link(object):
    def __init__(self, link_weight, link_property):
        self.link_weight = link_weight
        self.link_property = link_property

class Graph(object):
    def __init__(self, v_list):
        self.link_list = {}
        for vertex in v_list:
            self.link_list[vertex] = {}
            link = Link(1.0, 'self')
            self.add_link(vertex, vertex, link)
    def add_link(self, v1, v2, link):
        if v2 in self.link_list[v1]:
            if self.link_list[v1][v2].link_weight < link.link_weight:
                self.link_list[v1][v2] = link
        else:
            self.link_list[v1][v2] = link
    def bfs(self, v):
        neighbors = []
        neighbors.append(v)
        to_visit = deque([])
        to_visit.append(v)
        while len(to_visit) :
            v_cur = to_visit.pop()
            for v_neighbor in self.link_list[v_cur]:
                if not v_neighbor in neighbors:
                    neighbors.append(v_neighbor)
                    to_visit.append(v_neighbor)
        return neighbors

    def dijkstra(self, v, v_list):
        factor = {}
        prev = {}
        jump_P = {}
        jump_E = {}
        visited = {}
        factor[v] = 1
        prev[v] = v
        for v_neighbor in v_list:
            if v_neighbor in self.link_list[v]:
                factor[v_neighbor] = self.link_list[v][v_neighbor].link_weight
                prev[v_neighbor] = v
                if is_P(v):
                    jump_P[v_neighbor] = 1
                    jump_E[v_neighbor] = 0
                elif is_E(v):
                    jump_P[v_neighbor] = 0
                    jump_E[v_neighbor] = 1
                else:
                    print "error at line 67"
                    return
            else:
                factor[v_neighbor] = 0
                prev[v_neighbor] = 'unknow'
                jump_P[v_neighbor] = 0
                jump_E[v_neighbor] = 0
            visited[v_neighbor] = False
        visited[v] = True
        jump_P[v] = 0
        jump_E[v] = 0
        for i in range(len(v_list)-1):
            max_factor = 0
            u = v
            for v_neighbor in v_list:
                if not visited[v_neighbor] and factor[v_neighbor] > max_factor:
                    max_factor = factor[v_neighbor]
                    u = v_neighbor
            if u == v:
                print "error"
            else:
                visited[u] = 1
            for v_neighbor in self.link_list[u]:
                if not visited[v_neighbor]:
                    cur_factor = self.link_list[u][v_neighbor].link_weight
                    if factor[v_neighbor] < factor[u]*cur_factor:
                        factor[v_neighbor] = factor[u]*cur_factor
                        prev[v_neighbor] = u
                        if is_P(u):
                            jump_P[v_neighbor] = jump_P[u] + 1
                            jump_E[v_neighbor] = jump_E[u]
                        elif is_E(u):
                            jump_P[v_neighbor] = jump_P[u]
                            jump_E[v_neighbor] = jump_E[u] + 1
                        else:
                            print "error at line 102"
        return factor, prev, jump_P, jump_E

def is_P(v):
    return v[-1] == 'P'
def is_E(v):
    return v[-1] == 'E'

def load_vertex_E(vertex_fn, prop, v_list):
    vertex_file = open(vertex_fn, 'r')
    vertex_file.readline()
    blank_node_cnt = 0
    for line in vertex_file.readlines():
        vertex = line.strip().split(',')
        vertex[0] = vertex[0].strip('"')
        if vertex[9] == '':
            v = Vertex_E(vertex[0]+prop, prop, -1, 'E')
            blank_node_cnt = blank_node_cnt + 1
        else:
            v = Vertex_E(vertex[0]+prop, prop, float(vertex[9]), vertex[10])
        v_list[vertex[0]+prop] = v
   # print blank_node_cnt
    return v_list

def load_vertex_P(vertex_fn, prop, v_list):
    vertex_file = open(vertex_fn, 'r')
    vertex_file.readline()
    blank_node_cnt = 0
    for line in vertex_file.readlines():
        vertex = line.strip().split(',')
        vertex[0] = vertex[0].strip('"')
        if not vertex[1]:
            blank_node_cnt = blank_node_cnt + 1
        v = Vertex_P(vertex[0]+prop, prop, vertex[1])
        v_list[vertex[0]+prop] = v
   # print blank_node_cnt
    return v_list

def load_link(link_fn, graph):
    link_file = open(link_fn, 'r')
    link_file.readline()
    for line in link_file.readlines():
        link = line.strip().split(',')
        v1 = link[0].strip('"')+link[1].strip('"')
        v2 = link[2].strip('"')+link[3].strip('"')
        link_weight = float(link[4].strip('"'))
        link_prop_pos = link[5].strip('"')
        if link_prop_pos == 'FATHER':
            link_prop_neg = 'SON'
        elif link_prop_pos == 'SON':
            link_prop_neg = 'FATHER'
        elif link_prop_pos == 'OTHER':
            link_prop_neg = 'OTHER'
        else:
            print 'wrong link property!!!'
        if link_weight != 0:
            link_pos = Link(link_weight, link_prop_pos)
            link_neg = Link(link_weight, link_prop_neg)
            graph.add_link(v1, v2, link_pos)
            graph.add_link(v2, v1, link_neg)
    return graph


if __name__ == '__main__':
    v_list = {}
    v_list = load_vertex_E("./data/EINFOALL_ANON.csv", 'E', v_list)
    #print len(v_list)
    v_list = load_vertex_P("./data/PINFOALL_ANON.csv", 'P', v_list)
    #print len(v_list)

    graph = Graph(v_list)
    graph = load_link("./data/LINK_ANON.csv", graph)
    neighbors= graph.bfs('anon_S0E')
   # print len(neighbors)
   # for v in neighbors:
   #     print "(" + v + ")" #+ str(neighbors[v]) + ") "

    factors, prevs, jump_P, jump_E = graph.dijkstra('anon_S0E', neighbors)
    for v in factors:
        if is_E(v):
            print  v + "  " + str(factors[v]) + "  " +  str(jump_E[v])
