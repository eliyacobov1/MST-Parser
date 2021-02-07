from collections import namedtuple, defaultdict, Counter
import numpy as np
from nltk.corpus import dependency_treebank
from nltk import DependencyGraph
from typing import List

from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx

Arc = namedtuple('Arc', 'head tail weight')
MAX_SENT_LENGTH = 40


def create_iter(size):
    """
    return an integer iterator with range of the given param "size"
    """
    return iter(range(size))


class MST:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations
        corpus = dependency_treebank.parsed_sents()
        sep = int(len(corpus) * (9 / 10))
        self.train_set, self.test_set = corpus[:sep], corpus[sep:]
        self.word_vec, self.tag_vec = self.word_tag_vectors_init()
        self.num_words, self.num_tags = len(self.word_vec), len(self.tag_vec)
        self.cur_weights = Counter()
        self.aggregate_weights = Counter()  # sum of weights
    
    def word_tag_vectors_init(self):
        word_index = create_iter(len(self.train_set) * MAX_SENT_LENGTH)
        tag_index = create_iter(len(self.train_set) * MAX_SENT_LENGTH)
        word_vec, tag_vec = defaultdict(word_index.__next__), defaultdict(tag_index.__next__)
        for sent in self.train_set + self.test_set:  # add all of the words and tags
            for node in sent.nodes.values():
                word_vec[node["word"]]
                tag_vec[node["tag"]]
        return word_vec, tag_vec
    
    def feature_func(self, index1: int, index2: int, sent: DependencyGraph):
        """
        returns a tuple with the two indexes that are set to 1 after
        applying the feature function on the two nodes corresponding
        to the given indexes index1, index2 (Q2)
        """
        u, v = sent.nodes[index1], sent.nodes[index2]  # graph nodes of the given indexes
        return self.find_word_feature_index(u["word"], v["word"]), \
               self.find_tag_feature_index(u["tag"], v["tag"])
    
    def find_word_feature_index(self, w, w_tag):
        """
        returns the feature index corresponding to the given words w and w_tag
        """
        return self.word_vec[w] * self.num_words + self.word_vec[w_tag]
    
    def find_tag_feature_index(self, t, t_tag):
        """
        returns the feature index corresponding to the given tags t and t_tag
        """
        return self.num_words ^ 2 + self.tag_vec[t] * self.num_tags \
               + self.word_vec[t_tag]
    
    def calc_edge_score(self, index1: int, index2: int, sent: DependencyGraph):
        """
        calculates the score of the edge (u,v) in the fully connected
        graph of the given sentence where (u,v) are the nodes represented
        by the given indexes index1, index2
        """
        i, j = self.feature_func(index1, index2, sent)
        return self.cur_weights[i] + self.cur_weights[j]
    
    def features_sum(self, sent: DependencyGraph, arcs: List[tuple]) -> Counter:
        """
        returns the sum vector of the feature sum over the given arcs
        """
        return sum([Counter(self.feature_func(arc[0], arc[1], sent))
                    for arc in arcs], Counter())
    
    def max_spanning_tree(self, sent: DependencyGraph):
        """
        return the arcs of the maximal spanning tree for the given sentence
         (according to the current weight vector values)
        """
        # arcs of the fully connected graph composed
        # of the words in the given sentence "sent"
        arcs = [Arc(i, j, self.calc_edge_score(i, j, sent) * -1)
                for i in sent.nodes for j in sent.nodes if i != j and j != 0]
        mst = min_spanning_arborescence_nx(arcs, sink=None)
        return [(arc.head, arc.tail) for arc in mst.values()]
    
    def gold_standard_tree(self, sent: DependencyGraph):
        """
        return the arcs of the gold standard tree for the given sentence
        """
        root = sent.root["address"]
        edges = sent.nx_graph().edges
        # build the gold standard tree, edge weights are
        # of no significance and therefore are all set to 1
        return [(edge[1], edge[0]) for edge in edges] + [(0, root)]
        
    def perceptron_algorithm(self) -> Counter:
        """
        apply perceptron algorithm to learn the optimal weights for the MST (Q3)
        """
        for i in range(self.num_iterations):
            for j in range(len(self.train_set)):
                mst, gst = self.max_spanning_tree(self.train_set[j]), \
                           self.gold_standard_tree(self.train_set[j])
                mst_feature_sum = self.features_sum(self.train_set[j], mst)
                gst_feature_sum = self.features_sum(self.train_set[j], gst)
                gst_feature_sum.subtract(mst_feature_sum)  # the step size for this iteration
                self.cur_weights.update(gst_feature_sum)
                self.aggregate_weights.update(self.cur_weights)
                
        for key in self.aggregate_weights.keys():
            self.aggregate_weights[key] /= self.num_iterations * len(self.train_set)
        self.cur_weights = self.aggregate_weights
        return self.cur_weights
    
    def calc_attachment_score(self, sent):
        mst = self.max_spanning_tree(sent)
        gst = self.gold_standard_tree(sent)
        return len(set(mst).intersection(set(gst))) / len(gst)
        
    def evaluate(self):
        """
        evaluate the accuracy of the computed weights over the test set (Q4)
        """
        total_acc = 0
        for sent in self.test_set:
            total_acc += self.calc_attachment_score(sent)
        return total_acc / len(self.test_set)


M = MST(2)
M.perceptron_algorithm()
acc = M.evaluate()
print(acc)
