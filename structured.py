#!/usr/bin/env python
# -*- encoding: utf-8 -*-

LICENSE = """Copyright (c) 2012, Gereon Kaiping <anaphory@yahoo.de>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

import OpenBayes
from OpenBayes import BNet, BVertex, DirEdge, JoinTree

import logging
logging.getLogger(__name__).setLevel(logging.CRITICAL)
logging.getLogger("").setLevel(logging.CRITICAL)

import numpy

network = BNet('Number Guessing')

# Create a discrete node for all nodes with 2 states
numbers = network.add_v(BVertex('numbers', True, 10))
bigger_3 = network.add_v(BVertex('Is your number bigger than 3?', True, 2))
bigger_5 = network.add_v(BVertex('bigger_5', True, 2))
IsTheNumberBiggerThan7 = network.add_v(BVertex('IsTheNumberBiggerThan7', True, 2))
prime = network.add_v(BVertex('prime', True, 2))
even = network.add_v(BVertex('even', True, 2))
odd = network.add_v(BVertex('odd', True, 2))

network.add_e(DirEdge(len(network.e), numbers, bigger_3))
network.add_e(DirEdge(len(network.e), numbers, bigger_5))
network.add_e(DirEdge(len(network.e), numbers, IsTheNumberBiggerThan7))
network.add_e(DirEdge(len(network.e), numbers, even))
network.add_e(DirEdge(len(network.e), numbers, prime))

network.add_e(DirEdge(len(network.e), even, odd))

# Show the network
print network

# Initialize the distributions
network.InitDistributions()

# Set distributions for start nodes
numbers.setDistributionParameters([0.1]*10)

bigger_3.distribution[{'numbers':0}]=[1, 0]
bigger_3.distribution[{'numbers':1}]=[1, 0]
bigger_3.distribution[{'numbers':2}]=[1, 0]
bigger_3.distribution[{'numbers':3}]=[1, 0]
bigger_3.distribution[{'numbers':4}]=[0, 1]
bigger_3.distribution[{'numbers':5}]=[0, 1]
bigger_3.distribution[{'numbers':6}]=[0, 1]
bigger_3.distribution[{'numbers':7}]=[0, 1]
bigger_3.distribution[{'numbers':8}]=[0, 1]
bigger_3.distribution[{'numbers':9}]=[0, 1]

bigger_5.distribution[{'numbers':0}]=[1, 0]
bigger_5.distribution[{'numbers':1}]=[1, 0]
bigger_5.distribution[{'numbers':2}]=[1, 0]
bigger_5.distribution[{'numbers':3}]=[1, 0]
bigger_5.distribution[{'numbers':4}]=[1, 0]
bigger_5.distribution[{'numbers':5}]=[1, 0]
bigger_5.distribution[{'numbers':6}]=[0, 1]
bigger_5.distribution[{'numbers':7}]=[0, 1]
bigger_5.distribution[{'numbers':8}]=[0, 1]
bigger_5.distribution[{'numbers':9}]=[0, 1]

IsTheNumberBiggerThan7.distribution[{'numbers':0}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':1}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':2}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':3}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':4}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':5}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':6}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':7}]=[1, 0]
IsTheNumberBiggerThan7.distribution[{'numbers':8}]=[0, 1]
IsTheNumberBiggerThan7.distribution[{'numbers':9}]=[0, 1]

prime.distribution[{'numbers':0}]=[1, 0]
prime.distribution[{'numbers':1}]=[1, 0]
prime.distribution[{'numbers':2}]=[0, 1]
prime.distribution[{'numbers':3}]=[0, 1]
prime.distribution[{'numbers':4}]=[1, 0]
prime.distribution[{'numbers':5}]=[0, 1]
prime.distribution[{'numbers':6}]=[1, 0]
prime.distribution[{'numbers':7}]=[0, 1]
prime.distribution[{'numbers':8}]=[1, 0]
prime.distribution[{'numbers':9}]=[1, 0]

even.distribution[{'numbers':0}]=[0, 1]
even.distribution[{'numbers':1}]=[1, 0]
even.distribution[{'numbers':2}]=[0, 1]
even.distribution[{'numbers':3}]=[1, 0]
even.distribution[{'numbers':4}]=[0, 1]
even.distribution[{'numbers':5}]=[1, 0]
even.distribution[{'numbers':6}]=[0, 1]
even.distribution[{'numbers':7}]=[1, 0]
even.distribution[{'numbers':8}]=[0, 1]
even.distribution[{'numbers':9}]=[1, 0]

odd.distribution[{'even':0}]=[0, 1]
odd.distribution[{'even':1}]=[1, 0]

# Build a JoinTree
join_tree = JoinTree(network)

###
## Begin example inferences
###

# Give the network some evidence.
for number in xrange(10):
    join_tree.SetObs({"numbers": number})

    print
    print number
    print "Properties"

    for node in join_tree.BNet.v:
        print node, join_tree.Marginalise(node).cpt[1]


join_tree.SetObs({"odd": 1, bigger_3.name: 1})
print
print join_tree.Marginalise("prime")
join_tree.SetObs({})

join_tree.SetObs({"odd": 1})
print
print join_tree.Marginalise("even")
join_tree.SetObs({})

def xlogx(x):
    """Calculate x*log(x) for x array-like, continually extended to 0."""
    x = numpy.asarray(x)
    l = numpy.zeros_like(x)
    l[x>0] = numpy.log(x[x>0])
    return x * l

def entropy(Ps):
    """Calculate the 'Remaining Bit Rate' of the given probabilities.

    That is, the entropy measured in 0.693 bits (i.e. using log instead of lg_2)
    """
    Ps = numpy.asarray(Ps)
    Ps /= sum(Ps)
    Ps[numpy.isnan(Ps)] = 0
    return -sum(xlogx(Ps)) #/log(2), but that is a constant

def bayes(P_X_given_Y, P_Y, P_X):
    """Given P(X|Y), P(Y) and P(X), calculate P(Y|X)."""
    P_X_given_Y = numpy.asarray(P_X_given_Y)
    P_Y = numpy.asarray(P_Y)
    P_X = numpy.asarray(P_X)
    y = P_Y.view().reshape((1, -1))
    x = P_X.view().reshape((-1, 1))
    xy = P_X_given_Y.view().reshape((-1, y.shape[1]))
    yx = ((xy*y)/x).transpose()
    yx[numpy.isnan(yx)]=0
    yx.shape = (P_Y.shape+P_X.shape)
    return yx

def entropy_after_answer(network, evidence):
    """Given the probabilities of all answers to all questions, calculate the expected entropy after getting an answer to every question."""

    entropies = {}
    for name, node in network.BNet.v.iteritems():
        if name != "numbers" and evidence.get(name, None) is None:
            entropies[name] = 0
            ev = evidence.copy()
            network.SetObs(ev)
            for v, p in enumerate(network.Marginalise(name)):
                ev[name] = v
                network.SetObs(ev)
                entropies[name] += p*entropy(network.Marginalise("numbers"))
                network.SetObs(evidence.copy())

    print "Evidence will be reset to %r" % evidence
    network.SetObs(evidence.copy())

    return entropies

class Open20Q (object):
    #additional answer to be displayed
    additional = ["I don't know."]
    
    #conclusive question
    final = "Are you thinking of %s?"

    #generic answers
    Yes = {"Yes.": 1., "No.": 0.}
    No = {"Yes.": 0., "No.": 1.}

    #probability for a totally random question
    epsilon = 0.05

    #Always assume that this many games have led to the current
    #data. (This is something like a weight factor for the current
    #game.)
    n = 50

    def __init__(self, items, bayes_network):
        """
        """
        self.net = bayes_network
        for name, vertex in self.net.BNet.v.iteritems():
            if name != "numbers":
                vertex.answers = ["No.", "Yes."]

        self.items = items #The keys of the dictionary passed are the items to be identified.
        
        self.learner = OpenBayes.learning.SEMLearningEngine(
            self.net.BNet)
        
        self.cases = self.net.BNet.Sample(10000)

    def ask(self, question, answers):
        print question
        print " / ".join(answers)
        answer = raw_input()
        try:
            a = int(answer)
        except ValueError:
            try:
                a = answers.index(answer)
            except ValueError:
                b = [ans for ans in answers
                     if ans.lower().startswith(
                        answer.lower())]
                if len(b) == 1:
                    a = answers.index(b[0])
                else:
                    return self.ask(question, answers)
        return a
        
    def identify(self, max_questions=20):
        X = None
        ev = self.net.evidence.copy()
        not_solutions = set()
        for counter in xrange(max_questions):
            P_X = self.net.Marginalise("numbers").cpt
            for X in not_solutions:
                P_X[X] = 0
            if not P_X.any():
                X = None
                break
            print entropy(P_X)
            for i, item in enumerate(self.items):
                print item, P_X[i],
            print
            entr = entropy(P_X)
            entropies = entropy_after_answer(self.net, ev)
            print entropies
            if (entr < 1 or
                counter == max_questions-1 or 
                not entropies
                ):
                X = P_X.argmax()
                if self.ask(self.final%self.items[X], ["No.", "Yes."]):
                    break
                not_solutions.add(X)
                X = None
            else:
                if numpy.random.random()<self.epsilon:
                    print "Random Question:"
                    question = entropies.keys()[numpy.random.randint(len(entropies))]
                else:
                    print entropies
                    question = min(entropies, key=entropies.get)
                    print "Best Question:"
                q_answers = self.net.BNet.v[question].answers
                a = self.ask(question, q_answers)

                ev[question] = a
                self.net.SetObs(ev.copy())
                print "That is, ev is now %r." % ev

        if X is None:
            item_name = raw_input("What were you thinking of? - ")
            try:
                X = self.items.index(item_name)
                #add_question(item)
            except ValueError:
                self.add_item(ev, item_name)
        if X is not None:
            self.update(ev, X)

    def update(self, answers, item):
        answers["number"] = item
        self.cases.append(
            { node: answers.get(node, "?") 
              for node in self.net.BNet.v })

        self.learner.SEMLearning(self.cases, 1)

    def add_item(self, answers, item):
        self.items.append(item)
        root = self.net.BNet.v["numbers"]
        root.nvalues += 1
        root.distribution.sizes[0] += 1
        root.distribution.cpt = (
            numpy.concatenate((
                (1-1e-4) *
                root.distribution.cpt,
                [1e-4])))
        self.net.SetObs(answers.copy())
        for dependent in root.out_v:
            in_p = [edge._v[0]
                    for edge in dependent._e
                    ].index(root) + 1
            s = list(dependent.distribution.cpt.shape)
            s[in_p] = 1
            s = tuple(s)
            probs = 0.5*numpy.ones(s)
            
            dependent.distribution.cpt = numpy.concatenate((
                    dependent.distribution.cpt,
                    probs
                    ), axis=in_p)
        self.net = JoinTree(self.net.BNet)
        self.learner = OpenBayes.learning.SEMLearningEngine(
            self.net.BNet)

    def add_question(self, question, parents = ["numbers"]):
        network = self.net.BNet
        new = network.add_v(BVertex(question, nvalues=2))
        new.answers = ["No.", "Yes."]
        for parent in parents:
            parent_v = network.v[parent]
            network.add_e(
                DirEdge(len(network.e),
                        parent_v, new))

        new.InitDistribution()
        self.net = JoinTree(network)
        self.learner = OpenBayes.learning.SEMLearningEngine(
            network)
  
numbers = Open20Q(range(10), join_tree)

def run(n=1):
    for i in xrange(n):
        numbers.net.SetObs({})
 
        numbers.identify()

        if len(numbers.net.BNet.v) < 2*numpy.log(numbers.net.BNet.v["numbers"].nvalues)/numpy.log(2):
            numbers.add_question(raw_input("Gimme a new question!"))
