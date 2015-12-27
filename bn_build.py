#!usr/local/bin/python
#-*- coding:utf-8 -*-
import math
import pandas as pd
import numpy as np

class Vertex_E(object):
    def __init__(self, idx, prop, creditscore, rating):
        self.idx = idx
        self.prop = prop
        self.parent = {}
        self.e_parent_cnt = 0
        self.child = {}
        self.layer = -1 # 拓扑排序中的层数

        self.creditscore = creditscore
        self.rating = rating

        self.norm_score = None
        self.root_marginal_dist = None
        self.global_lpd = None
        self.global_marginal_dist = None
        self.global_new_score = None

        self._no_score_in_root = 1 # 节点是否为root中原始分缺失节点，默认是

    def set_parent(self,p,w):
        p_id = p.idx
         # if multi edges exist, choose the one with largest weight
        if self.parent.has_key(p_id) and self.parent[p_id][1] >= w:
            pass
        else:
            self.parent[p_id] = [p,w]
            if p_id[-1] == 'E':
                self.e_parent_cnt += 1

    def set_child(self,c,w):
        c_id = c.idx
        # if multi edges exist, choose the one with largest weight
        if self.child.has_key(c_id) and self.child[c_id][1] >= w:
            pass
        else:
            self.child[c_id] = [c,w]

    def print_vertex(self):
        print "=====","vertex name:",self.idx,"====="
        print "layer in topSort:",self.layer
        print "Parents:", ",".join([v + " " + str(self.parent[v][1]) for v in self.parent])
        print "No. of E parents:",self.e_parent_cnt
        print "Child:", ",".join([v + " " + str(self.child[v][1]) for v in self.child])

    def set_root_marginal_dist(self):
        prob = sigmoid(self.norm_score) # 更改此函数替换分数与概率的对应
        self.root_marginal_dist = pd.DataFrame({'P':[prob,1-prob],self.idx:[0,1]},columns = ['P',self.idx])

    def set_global_lpd(self):
        print "generate lpd for node:",self.idx
        parent_id_list = []
        parent_weight_list = []
        parent_df_list = []
        p_parent_weight = 0

        for k in self.parent:
            if k[-1] == 'P':
                p_parent_weight += self.parent[k][1]
                continue
            if not self.parent[k][0]._no_score_in_root:
                parent_id_list.append(k)
                parent_weight_list.append(self.parent[k][1])
                parent_df_list.append(self.parent[k][0].global_marginal_dist)

        if p_parent_weight > 0 and self.creditscore > -1:
            # 在e父亲节点后加入p父亲节点的加总，名称为P_parents,权重为p节点的加总，其df_list为当前节点的原始边缘概率
            parent_id_list.append('P_parents')
            parent_weight_list.append(p_parent_weight)
            p_pa_marginal_df = self.root_marginal_dist.copy()
            p_pa_marginal_df.columns = ['P','P_parents']
            parent_df_list.append(p_pa_marginal_df)

        num_of_val_parents = len(parent_id_list)
        print "num of parents:",self.e_parent_cnt,num_of_val_parents
        if num_of_val_parents > 11:
            print "too many parents! parent number:",num_of_val_parents
            return self.root_marginal_dist,[]

        # 将边权重归一化
        sum_w = sum(parent_weight_list) * 1.0
        parent_weight_list = map(lambda w: w / sum_w,parent_weight_list)
        # print parent_id_list,parent_weight_list,parent_df_list

        # 求得lambda
        para_list = np.array(cal_parameter_list(parent_id_list,parent_weight_list,parent_df_list))
        # print para_list

        # Noisy-Or计算lpd条件概率分布
        lpd_cols = ['P',self.idx] + parent_id_list
        lpd_df = generate_df(lpd_cols)

        # 目前认为若父节点均不出问题，则不会出问题，即norisy-or中lambda0 = 0
        # 此处 leak_prob = P(y=0|pa(y) = 0)
        if self.creditscore > -1 and num_of_val_parents > 1:
            # leak_prob = self.root_marginal_dist[self.root_marginal_dist[self.idx] == 0]['P']
            leak_prob = 0.9
        else:
            # 对无原始分及父亲只有一个的节点，lambda0 = 1-leak_prob = 0；父亲只有一个的节点，其para为1，若存在leak则客观上降低其变好的概率
            leak_prob = 1
        # leak_prob = 1
        lpd_df.ix[0]['P'] = leak_prob
        for i in range(1,lpd_df.shape[0]):
            if lpd_df.ix[i][2:].sum() == 0 and lpd_df.ix[i][self.idx] == 1:
                lpd_df.ix[i]['P'] = 1 - leak_prob
                continue
            # 对于父节点数量为1的，直接设定；用reduce计算
            if lpd_df.ix[i][self.idx] == 0:
                lpd_df.ix[i]['P'] = (1 - para_list[0]) if num_of_val_parents == 1 else reduce(lambda a,b:a*b,[f for f in lpd_df.ix[i][2:] * (1 - para_list) if f > 0])
            else:
                lpd_df.ix[i]['P'] = 1 - (1 - para_list[0]) if num_of_val_parents == 1 else 1 - reduce(lambda a,b:a*b,[f for f in lpd_df.ix[i][2:] * (1 - para_list) if f > 0])

        return lpd_df,parent_df_list

    def set_global_marginal_dist(self,is_root):
        self._no_score_in_root = 0
        if is_root:
            self.global_lpd = self.root_marginal_dist
            self.global_marginal_dist = self.root_marginal_dist
        else:
            lpd,parent_df_list = self.set_global_lpd()
            if len(parent_df_list) > 0:
                self.global_lpd = lpd
                joint_df = self.global_lpd.copy()
                for p_df in parent_df_list:
                    joint_df = multiplyDf(joint_df,p_df)
                marginal_df = joint_df.groupby(self.idx).sum()['P'].reset_index() # 层次化索引转为列
                marginal_df = marginal_df.reindex_axis(['P',self.idx],1) # 调整列顺序
                self.global_marginal_dist = marginal_df

            else:
                print "no lpd computed because of too many parents"
                self.global_marginal_dist = lpd

class Vertex_P(object):
    def __init__(self, idx, prop, cerno):
        self.idx = idx
        self.prop = prop
        self.cerno = cerno
        self.parent = {}
        self.child = {}
class Link(object):
    def __init__(self, link_weight, link_property):
        self.link_weight = link_weight
        self.link_property = link_property

class Graph(object):
    def __init__(self, v_list):
        self.vertex_list = v_list
        self.root_list = []
        self._min_score = 0 # 记录原始评分的最大最小分值，依次转换求出的边缘概率，得到新的分数
        self._max_score = 0

    def set_edge(self,parent,child,w,link_void_e_cnt):
        c_v = self.vertex_list[child]
        if parent[-1] == 'P':
            p_v = Vertex_E(parent,'P',-2,None) # 生成虚的P节点，根据prop确定是否为E
            c_v.set_parent(p_v,w)
        elif (parent[-1] == 'E') and (self.vertex_list[parent].creditscore > -1): # 忽略信用分缺失的父节点边，共661条
            p_v = self.vertex_list[parent]
            p_v.set_child(c_v,w)
            c_v.set_parent(p_v,w)
        else:
            link_void_e_cnt += 1
        return link_void_e_cnt

    def get_root(self):
        self.root_list = []
        for v_id in self.vertex_list:
            v = self.vertex_list[v_id]
            if v.prop == 'E' and v.e_parent_cnt == 0:
                v.layer = 0
                self.root_list.append(v)

    # use topSort to allocate layers for vertices
    def top_sort(self):
        vertex_layers = {}
        layer = 0
        vertex_layers[layer] = self.root_list
        visited_list = [] # stores vertices already visited
        print "Layer No:",layer,"num of v:",len(vertex_layers[layer])

        while len(vertex_layers[layer]) != 0:
            visited_list.extend(vertex_layers[layer])
            layer += 1
            vertex_layers[layer] = []
            for v in vertex_layers[layer - 1]:
                if len(v.child) > 0:
                    for chv_id in v.child:
                        chv = v.child[chv_id][0]
                        ch_parent_set = set([chv.parent[chp][0] for chp in chv.parent if chp[-1] == 'E'])
                        if ((ch_parent_set & set(visited_list)) == ch_parent_set) and (chv not in vertex_layers[layer]):
                            vertex_layers[layer].append(chv)
                            chv.layer = layer
            print "Layer No:",layer,"num of v:",len(vertex_layers[layer])
        return vertex_layers

    def normalize_score(self):
        score_list = [self.vertex_list[v].creditscore for v in self.vertex_list if self.vertex_list[v].creditscore > -1]
        min_score = min(score_list)
        max_score = max(score_list)
        max_min = max_score - min_score
        for v_id in self.vertex_list:
            if self.vertex_list[v_id].creditscore > -1:
                self.vertex_list[v_id].norm_score = (self.vertex_list[v_id].creditscore - min_score) / max_min
                self.vertex_list[v_id].set_root_marginal_dist()
            else:
                self.vertex_list[v_id].norm_score = self.vertex_list[v_id].creditscore
        self._min_score = min_score
        self._max_score = max_score

    def build_global_bn(self,layer_subgraph):
        print "==== start to build global bayesian model ===="
        subgraph_cnt = len(layer_subgraph) - 1
        print subgraph_cnt
        for i in range(subgraph_cnt):
            cur_vertex_list = layer_subgraph[i]
            for cur_v in cur_vertex_list:
                if i == 0:
                    if cur_v.creditscore == -1: continue # 对于根节点中无分数的点，直接跳过
                    cur_v.set_global_marginal_dist(1)
                    # print cur_v.idx,cur_v.creditscore
                    # print cur_v.global_marginal_dist
                    continue
                print "layer:",i,cur_v.idx
                cur_v.set_global_marginal_dist(0)
        print "==== global bayesian model built successfully! ===="

    def convert_to_new_score(self):
        print "== convert to new score... =="
        for v_id in self.vertex_list:
            if v_id[-1] == 'P': continue
            v = self.vertex_list[v_id]
            if v._no_score_in_root:
                v.global_new_score = -1
                continue
            if v.layer == 0:
                v.global_new_score = v.creditscore
            else:
                marginal_df = v.global_marginal_dist
                p = marginal_df[marginal_df[v.idx] == 0]['P']
                norm_new_score = float(np.log(p / (1-p)))
                new_score = norm_new_score * (self._max_score - self._min_score) + self._min_score
                v.global_new_score = round(new_score,2)
            # print v.idx,v.creditscore,v.global_new_score
        print "== new scores generated =="

    def print_bn_result(self):
        print "node,layer,original_score,new_score,diff(new-origin)"
        for v_id in self.vertex_list:
            if v_id[-1] == 'P': continue
            node = self.vertex_list[v_id]
            node_list = [node.idx,node.layer,node.creditscore,node.global_new_score]
            if node.creditscore == -1:
                node_list.append("na")
            else:
                node_list.append(node.global_new_score - node.creditscore)
            print ",".join([str(i) for i in node_list])

    def get_local_structure(self, query_node_list):
        for v in self.vertex_list:
            if v in query_node_list:
                self.vertex_list[v].print_vertex()

def sigmoid(x):
    return 1 / (1 + math.e ** (-1 * x))

def is_P(v):
    return v[-1] == 'P'

def is_E(v):
    return v[-1] == 'E'

def bin_to_int_list(bin_val,length):
    string = str(bin_val)[2:]
    string = '0' * (length - len(string)) + string
    return [int(i) for i in string]

def load_vertex_E(vertex_fn, prop, v_list):
    vertex_file = open(vertex_fn, 'r')
    vertex_file.readline()
    blank_node_cnt = 0
    e_node_cnt = 0
    for line in vertex_file.readlines():
        e_node_cnt += 1
        vertex = line.strip().split(',')
        vertex[0] = vertex[0].strip('"')
        if vertex[9] == '':
            v = Vertex_E(vertex[0]+prop, prop, -1, 'E')
            blank_node_cnt = blank_node_cnt + 1
        else:
            v = Vertex_E(vertex[0]+prop, prop, float(vertex[9]), vertex[10])
        v_list[vertex[0]+prop] = v
    # print blank_node_cnt
    print "Total enterprise node:",e_node_cnt
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
    link_cnt = 0
    link_other_cnt = 0 # 无向边计数
    link_ep_cnt = 0 # e->p边计数
    link_void_e_cnt = 0 # e->e边父节点无原始分边计数
    for line in link_file.readlines():
        link_cnt += 1
        link = line.strip().split(',')
        v1 = link[0].strip('"') + link[1].strip('"')
        v2 = link[2].strip('"') + link[3].strip('"')
        link_weight = float(link[4].strip('"'))
        link_prop_pos = link[5].strip('"')

        # 跳过无向边，无向边共3049条
        if link_prop_pos == 'OTHER':
            link_other_cnt += 1
            continue

        if v1[-1] == 'P' and v2[-1] == 'P':
            print line
            continue

        # 只保留儿子为E的边
        if link_weight != 0:
            if link_prop_pos == 'FATHER' and v2[-1] == 'E':
                link_void_e_cnt = graph.set_edge(v1,v2,link_weight,link_void_e_cnt)
            elif link_prop_pos == 'SON' and v1[-1] == 'E':
                link_void_e_cnt = graph.set_edge(v2,v1,link_weight,link_void_e_cnt)
            else:
                link_ep_cnt += 1 # 观察发现无E->P的有向边
                print 'son is p',line
    print link_cnt,link_other_cnt,link_ep_cnt,link_void_e_cnt # 45831 3049 0 659
    return graph

def cal_parameter_list(id_list,weight_list,df_list):
    para_list = []
    total_denominator = sum([df_list[j].ix[0]['P'] * weight_list[j] for j in range(len(df_list))])
    for i in range(len(id_list)):
        prob_1 = df_list[i].ix[1]['P']
        numerator =  prob_1 * weight_list[i]
        denominator = total_denominator + numerator - (1 - prob_1) * weight_list[i]
        para_list.append(numerator / denominator)
        assert numerator / denominator > 0
    return para_list

def generate_df(cols):
    num_of_rows = 2 ** (len(cols) - 1)
    rows_list = [[0.0] + bin_to_int_list(bin(i),len(cols) - 1) for i in range(num_of_rows)]
    df = pd.DataFrame(index = range(num_of_rows),columns = cols)
    for i in range(num_of_rows):
        df.loc[i] = rows_list[i]
    df['P'].astype(float)
    return df

def multiplyDf(df1,df2):
    # print df1,df2
    mutualNodes = list(set(df1.columns) & set(df2.columns) - set('P'))
    mergedCol = ['P'] + list((set(df1.columns) | set(df2.columns)) - set('P'))
    dfm = pd.merge(df1,df2,on = mutualNodes,suffixes = ('_1','_2'))
    dfm['P'] = dfm['P_1'] * dfm['P_2']
    dfm = dfm.drop(['P_1','P_2'],1)
    dfm = dfm.reindex_axis(mergedCol,axis = 1)
    # print dfm
    return dfm

if __name__ == '__main__':

    v_list = {}
    v_list = load_vertex_E("./data/EINFOALL_ANON.csv", 'E', v_list)
    # v_list = load_vertex_P("./data/PINFOALL_ANON.csv", 'P', v_list)
    graph = Graph(v_list)
    graph = load_link("./data/LINK_ANON.csv", graph)
    # for v in graph.vertex_list:
    #     graph.vertex_list[v].print_vertex()
    graph.normalize_score()
    graph.get_root()
    print "root cnt:",len(graph.root_list)

    layer_subgraph = graph.top_sort()
    # query_node_list = ["anon_S516E"]
    # graph.get_local_structure(query_node_list)
    graph.build_global_bn(layer_subgraph)
    graph.convert_to_new_score()
    graph.print_bn_result()
