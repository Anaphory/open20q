#!/usr/bin/env python3.3
# -*- encoding: utf-8 -*-

# According to the ideas of http://lists.canonical.org/pipermail/kragen-tol/2010-March/000912.html
            
import numpy
import operator
from collections import OrderedDict, defaultdict

from functools import reduce
def normalize(probabilities):
    " rescale `probabilities` so that its entries add up to 1 "
    return probabilities/probabilities.sum()

def c(constant): return lambda *argv, **kwargs: constant
id = lambda x:x

class NaiveBayes:
    def __init__(self,
                 cpd_matrix,
                 classes=None,
                 features=None,
                 values=None,
                 prior=None):
        self.cpd = numpy.asarray(cpd_matrix)
        self.__set_features(features)
        self.__set_values(values, classes)
        self.__set_prior(prior)

        self.del_evidence()

    def __set_prior(self, prior):
        if prior is None:
            prior = normalize(numpy.ones(self.cpd.shape[1]))
        self.prior = prior

    def __set_features(self, features=None):
        try:
            features.items()
        except AttributeError:
            if features is None:
                features = range(self.cpd.shape[0])
            features = {f: [i] for i, f in enumerate(features)}
        self.features = features

    def __set_values(self, values=None, classes=None):
        if values is None:
            values = {}
        for feature, cpd_indices in self.features.items():
            if len(cpd_indices) == 1:
                self.cpd = numpy.vstack(
                    (self.cpd,
                     [1-self.cpd[cpd_indices[0]]]))
                cpd_indices.append(len(self.cpd)-1)
                values[feature] = ("Yes", "No")
            else:
                values.setdefault(feature,
                                  range(len(cpd_indices)))
        self.values = values

        if classes is None:
            classes = range(self.cpd.shape[1])
        self.values["class"] = classes

    def belief(self, node):
        if node == "class":
            posterior = self.prior*self.class_evidence
            ev = self.evidence
            given = numpy.isfinite(ev)
            if given.any():
                yes = (ev[given] * self.cpd[given].T).T
                no = ((1-ev[given]) * (1-self.cpd[given]).T).T
                lambdas = yes+no
                if numpy.isfinite(ev).any():
                    for l in lambdas:
                        posterior *= l
            return normalize(posterior)
        else:
            cpds = self.features[node]
            belief = numpy.dot(self.cpd[cpds],
                          self.belief("class"))
            if numpy.isfinite(self.evidence[cpds]).all():
                return belief*self.evidence[cpds]
            else:
                return belief

    def set_evidence(self, node, value):
        if node == "class":
            self.class_evidence = value
        else:
            cpd_indices = self.features[node]
            self.evidence[cpd_indices] = value

    def get_evidence(self, node):
        if node == "class":
            return self.class_evidence
        else:
            cpd_indices = self.features[node]
            return self.evidence[cpd_indices]

    def del_evidence(self):
        self.evidence = numpy.array(
            [numpy.nan]*self.cpd.shape[0])
        self.class_evidence = numpy.ones(self.cpd.shape[1])

    def add_class(self, name, epsilon=1e-2):
        self.values["class"].append(name)
        new_cpd = numpy.hstack((
            self.cpd,
            numpy.zeros((self.cpd.shape[0],1))))
        for f, i in self.features.items():
            b = self.belief(f)
            if numpy.isfinite(b).all():
                new_cpd[i, -1] = (1-epsilon)*b+epsilon*(1-b)
            else:
                new_cpd[i, -1] = .5
        self.cpd=new_cpd
        self.prior=normalize(numpy.hstack((
            self.prior,
            (epsilon,))))
        self.del_evidence()
            
    def add_feature(self, name, values=["Yes","No"]):
        lines = len(values)
        cpd = numpy.vstack(
            (self.cpd,
             (numpy.ones((lines, self.cpd.shape[1]))/lines)))
        self.features[name] = list(range(len(cpd)-lines, len(cpd)))
        self.values[name] = values
        self.cpd = cpd
        self.del_evidence()

    def update_from_evidence(self, name, epsilon=1e-2):
        item = self.values["class"].index(name)
        for f, i in self.features.items():
            b = self.belief(f)
            if numpy.isfinite(b).all():
                self.cpd[i, item] = (
                    self.cpd[i, item]*(1-epsilon) 
                    + b[1]*epsilon)
        self.prior = self.prior*(1-epsilon) 
        self.prior[item] += epsilon
            
class RandomNaiveBayes (NaiveBayes):
    def __init__(self,
                 cpd_matrix,
                 n_classifiers=20,
                 classes=None,
                 features=None,
                 values=None,
                 prior=None):

        self.__set_features(features)
        self.__set_values(values, classes)

        self.classifiers = []
        for n in range(n_classifiers):
            features = {}
            cpd = []
            for f, cpd_lines in self.features.items():
                if numpy.random.random()<0.901:
                    features[f] = []
                    for c in cpd_lines:
                        cpd.append(self.cpd[c])
                        features[f].append(len(cpd)-1)
            self.classifiers.append(NaiveBayes(
                cpd_matrix=cpd, 
                classes=self.values["class"][:], 
                features=features,
                prior=self.prior) )
        self.del_evidence()
        
    def __wrap_method(m, acc=lambda new,x:x, init=None, ret=id):
        def wrapped(self, *args, **kwargs):
            x = init
            for c in self.classifiers:
                try: x = acc(m(c, *args, **kwargs), x)
                except KeyError: pass
            return ret(x)
        return wrapped

    belief = __wrap_method(
        NaiveBayes.belief, lambda new, x: (x[0]+new, x[1]+1),
        (0,0), lambda x: x[0]/x[1]) 

    set_evidence = __wrap_method(NaiveBayes.set_evidence)
    del_evidence = __wrap_method(NaiveBayes.del_evidence)
    get_evidence = __wrap_method(
        NaiveBayes.get_evidence, lambda new, x:
        (x[0]+new, x[1]+1), (0,0), lambda x: x[0]/x[1])
    update_from_evidence = __wrap_method(
        NaiveBayes.update_from_evidence)

    def add_class(self, name, epsilon=1e-2):
        self.values["class"].append(name)
        for c in self.classifiers: c.add_class(name, epsilon)

    def add_feature(self, name):
        self.features[name] = None
        for c in self.classifiers:
            if numpy.random.random()<0.901: c.add_feature(name)

# ==== Pandas Testing ====
import pandas
Dx = pandas.DataFrame({"x": [False,True], "p": [0.9,0.1]})
Dx.set_index(["x"], inplace=True)
Dxy = pandas.DataFrame({"x": [False,True,False,True], "y": [False,False,True,True], "p": [0.2,0.8,0.9,0.1]})
Dxy.set_index(["x","y"], inplace=True)
Dxz = pandas.DataFrame({"x": [False,False,False,True], "z": [0,1,2,0], "p": [0.2,0.4,0.4,1]})
Dxz.set_index(["x","z"], inplace=True)
# ==== (End) ====

class Factor:
    def __init__(self, cpd_table, name=None):
        """ Build a CPD lookup function with name `name` from the DataFrame cpd_table """
        self.name = name
        self.cpd = cpd_table
        self.children = []
        self.parents = []
    @property
    def argspec(self):
        return self.cpd.index.names
    @property
    def __name__(self):
        return self.name
    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            return self.cpd.get_value(args[0], "p")
        else:
            return self.cpd.get_value(tuple(args), "p")
    @property
    def domain(self):
        try:
            return self.cpd.index.levels[-1]
        except AttributeError:
            return self.cpd.index

from bayesian.bbn import BBN, BBNNode
import bayesian
class TreeAugmentedNaiveBayes (BBN, NaiveBayes):
    """A class for Tree (or Forest) Augmented Naive Bayes Classifiers. It
 follows the same interface as NaiveBayes above (it is a subclass with
 most methods overwritten), but is a `bayes.BBN` under the hood (and
 therefore a subclass of that).
    """ 
    def __init__(self,
                 factors,
                 classes=None,
                 features=None,
                 values=None,
                 prior=None):
        variables = set()
        domains = {}
        variable_nodes = {}
        factor_nodes = {}
        for factor in factors:
            # factor: .argspec, .__call__, .__name__
            variables.update(factor.argspec)
            factor_nodes[factor.name] = factor
            for parent in factor.argspec:
                factor.parents.append(factor_nodes[parent])
            
        
        for factor in factor_nodes.values():
            for parent in factor.argspec:
                if parent != factor.name:
                    bayesian.bbn.connect(factor_nodes[parent], factor)
            domains[factor.name] = factor.func.domain
        BBN.__init__(self, factor_nodes, name="Classifier")
        self.domains = domains
        self.evidence = {}

    @property
    def values(self):
        return self.domains

    def belief(self, node):
        # Modified from BBN.query
        node = self.vars_to_nodes[node]
        jt = self.build_join_tree()
        assignments = jt.assign_clusters(self)
        jt.initialize_potentials(assignments,
                                 self,
                                 self.evidence)

        jt.propagate()
        marginals = dict()
        normalizers = defaultdict(float)

        for k, v in jt.marginal(node).items():
            # For a single node the
            # key for the marginal tt always
            # has just one argument so we
            # will unpack it here
            marginals[k[0]] = v
            # If we had any evidence then we
            # need to normalize all the variables
            # not evidenced.
            if self.evidence:
                normalizers[k[0][0]] += v

        if self.evidence:
            for k, v in marginals.items():
                if normalizers[k[0]] != 0:
                    marginals[k] /= normalizers[k[0]]

        return [marginals[(node.name, value)]
                for value in self.domains[node.name]]

    @property
    def features(self):
        return set(self.vars_to_nodes)-set(["class"])
    def set_evidence(self, node, value):
        self.evidence[node] = value
    def get_evidence(self, node):
        return self.evidence.get(node)
    def del_evidence(self):
        self.evidence = {}
    def add_class(self, name, epsilon=1e-2): raise NotImplementedError
    def add_feature(self, name, values=["Yes","No"]): raise NotImplementedError
    def update_from_evidence(self, name, epsilon=1e-2): raise NotImplementedError

knowledge = NaiveBayes(
    [[1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
     [0,.1,.8, 1,.9, 1,.8,.1, 0, 0],
     [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]],
    classes=["1","2","3","4","5","6","7","8","9","0"],
    features={"Is X<=5?":[0], "Is X>5?":[1], "Is X odd?":[2], "Is X even?":[3], "Is X<=1?":[4], "Is X<=7?":[5], "Is X close to 5?":[6], "Is X prime?":[7], "What is X mod 3?":[8,9,10]},
    values={"What is X mod 3?":["0","1","2"]})

def cpd_matrix_to_factors(
                 cpd_matrix,
                 classes=None,
                 features=None,
                 values=None,
                 prior=None):
    self_cpd = numpy.asarray(cpd_matrix)
    if prior is None:
        prior = normalize(numpy.ones(self_cpd.shape[1]))
    self_prior = prior

    try:
        features.items()
    except AttributeError:
        if features is None:
            features = range(self_cpd.shape[0])
        features = {f: [i] for i, f in enumerate(features)}
    self_features = features

    if values is None:
        values = {}
    for feature, cpd_indices in self_features.items():
        if len(cpd_indices) == 1:
            self_cpd = numpy.vstack(
                (self_cpd,
                 [1-self_cpd[cpd_indices[0]]]))
            cpd_indices.append(len(self_cpd)-1)
            values[feature] = ("Yes", "No")
        else:
            values.setdefault(feature,
                              range(len(cpd_indices)))
    self_values = values

    if classes is None:
        classes = range(self_cpd.shape[1])
    self_values["class"] = classes

    factors = [
        Factor(pandas.DataFrame([
            {"class": c, "p": p}
            for c, p in zip(self_values["class"],
                            prior)]).set_index(["class"]),
               "class")]
    for feature, rows in self_features.items():
        factors.append(Factor(pandas.DataFrame([
            {"class":c, feature: v, "p": p[i]}
            for v, p in zip(self_values[feature],
                            self_cpd[rows])
            for i, c in enumerate(self_values["class"])])
                       .set_index(["class",feature]),
                              feature))
    for cl in self_values["class"]:
        factors.append(Factor(pandas.DataFrame([
            {"class": c, cl: boolean, "p": float((c==cl)==boolean)}
             for c in self_values["class"]
             for boolean in [True, False]]).set_index(["class",cl]), cl))

    return factors

knowledge = TreeAugmentedNaiveBayes(map(BBNNode, cpd_matrix_to_factors([[1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
     [0,.1,.8, 1,.9, 1,.8,.1, 0, 0],
     [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]],
    classes=["1","2","3","4","5","6","7","8","9","0"],
    features={"Is X<=5?":[0], "Is X>5?":[1], "Is X odd?":[2], "Is X even?":[3], "Is X<=1?":[4], "Is X<=7?":[5], "Is X close to 5?":[6], "Is X prime?":[7], "What is X mod 3?":[8,9,10]},
    values={"What is X mod 3?":["0","1","2"]})))  
        
epsilon = 0.1
questions = (int(numpy.log(len(knowledge.values["class"]))/numpy.log(2) * (1/(1-epsilon)))+2)
history = 100

def xlogx(x):
    x = numpy.asarray(x)
    l = numpy.zeros_like(x)
    l[x>0] = numpy.log(x[x>0])
    return x * l

def entropy(ps):
    return -xlogx(ps).sum()


class Interface:
    def __init__(self):
        pass
    def pose_question(self, question, answers):
        print(question)
        print(", ".join("{:s}: {:d}".format(str(answer), i)
                        for i, answer in enumerate(sorted(answers))))
        x = input("> ")
        x = int(x)
        x = sorted(answers)[x]
        print(x, type(x) if type(x)!=str else "")
        return x
    def pose_final_question(self, item):
        return self.pose_question("Is it {:s}?".format(item), [False, True])
    def pose_alternative(self):
        return input("What then? ")
    def win(self):
        print("Yay, I win. Another!")


interface = Interface()

def decide_question(knowledge, epsilon=None, force_final=False):
    if epsilon is None:
        epsilon = 0.1

    if force_final or entropy(knowledge.belief("class"))<1:
        p = numpy.argmax(knowledge.belief("class"))
        return knowledge.values["class"][p]
        
    questions = [
        f for f in knowledge.features 
        if knowledge.get_evidence(f) is None]
    if numpy.random.random()<epsilon:
        #With an epsilon chance,
        #ask a random question
        return numpy.random.choice(questions)
        #Otherwise, ask the most relevant question.
    entropies_after_answer = {}
    for question in questions:
        entropies_after_answer[question] = 0
        p_answers = knowledge.belief(question)
        for answer, p in zip(knowledge.values[question],
                             p_answers):
            if p > 0:
                knowledge.set_evidence(question, answer)
                entropies_after_answer[question] += p * entropy(
                    knowledge.belief("class"))
        del knowledge.evidence[question]
        if not numpy.isfinite(entropies_after_answer[question]):
            entropies_after_answer[question] = numpy.inf
    return min(
        entropies_after_answer,
        key = entropies_after_answer.get)
            
def run(questions, knowledge, interface):
    knowledge.del_evidence()
    for i in range(questions):
        if i>=questions-1:
            question = decide_question(knowledge, force_final=True)
        else:
            question = decide_question(knowledge)
            
        if question in knowledge.values["class"]:
            answer = interface.pose_final_question(question)
        else:
            answers = knowledge.values[question]
            answer = interface.pose_question(question, answers)
        knowledge.set_evidence(question, answer)
        if question in knowledge.values["class"] and answer:
            item = question
            interface.win()
            break
    else:
        item = interface.pose_alternative()

    if item in knowledge.values["class"]:
        knowledge.update_from_evidence(item)
    else:
        knowledge.add_class(item)
    
    knowledge.del_evidence()
        
  
if __name__ == "__main__":
    while 1:
        run(questions, knowledge, interface)
