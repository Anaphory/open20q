# According to the ideas of http://lists.canonical.org/pipermail/kragen-tol/2010-March/000912.html

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


import numpy
from collections import OrderedDict

def xlogx(x):
    x = numpy.asarray(x)
    l = numpy.zeros_like(x)
    l[x>0] = numpy.log(x[x>0])
    return x * l

def rbr(Ps):
    # remaining entropy in bits
    Ps = numpy.asarray(Ps)
    Ps /= sum(Ps)
    Ps[numpy.isnan(Ps)] = 0
    return -sum(xlogx(Ps)) #/log(2), but that is a constant

def bayes(P_X_given_Y, P_Y, P_X):
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

def rbr_after_answer(P_QA, P_XQA):
    # There might be a more transparent formula.
    return sum((P_QA * rbr(P_XQA)).transpose())

class open20q (object):
    additional = ["I don't know."]
    final = "Are you thinking of %s?"
    Yes = {"Yes.": 1., "No.": 0.}
    No = {"Yes.": 0., "No.": 1.}
    epsilon = 0.05
    n = 50

    def __init__(self, items):
        """
        Parameters:
        items   # {"Thing 1":
                    {"Question 1":
                      {"Answer 1.1": P(A1.1|T1),
                       ...}
                     ...}
                   ...}
        """
        self.questions = OrderedDict()
        max_answers = 2
        self.items = items.keys()
        for questions in items.values():
            for (question, answers) in questions.iteritems():
                known_answers = self.questions.setdefault(question, self.additional[:])
                for answer in answers:
                    if answer not in known_answers:
                        known_answers.append(answer)
                max_answers = max(max_answers, len(known_answers))
        self.connections = numpy.zeros((len(self.questions), max_answers, len(items)))
            #Use a sparse tensor or database implementation later.
        for i, item in enumerate(self.items):
            for q, (question, answers) in enumerate(self.questions.iteritems()):
                for a in xrange(len(answers)):
                    self.connections[q, a, i] = items.get(item, {}).get(question, {}).get(answers[a], 0)
        self.item_frequencies = numpy.ones(len(self.items))
        self.max_answers = max_answers

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
                a = 0
        return a
        
    def identify(self, max_questions=20):
        P_X = self.item_frequencies
        answers = {}
        X = None
        for counter in xrange(max_questions):
            print rbr(P_X)
            for i, item in enumerate(self.items):
                print item, P_X[i],
            print

            P_QA = numpy.tensordot(self.connections, P_X, 1)
            P_XQA = bayes(self.connections, P_X, P_QA)
            rbrs = rbr_after_answer(P_QA, P_XQA)
            for q in answers.keys():
                rbrs[q] = numpy.inf
            q = rbrs.argmin()
            entropy = rbr(P_X)
            if entropy < 1 or counter == max_questions-1:
                X = P_X.argmax()
                if self.ask(self.final%self.items[X], ["No.", "Yes."]):
                    break
                P_X[X] = 0
                X = None
                if entropy == 0:
                    break
            else:
                if numpy.random.random()<self.epsilon:
                    print "Random Question:"
                    q = numpy.random.randint(len(self.questions))
                else:
                    print "Best Question:"
                question, q_answers = self.questions.items()[q]
                a = self.ask(question, q_answers)
                answers[q] = numpy.zeros(3)
                answers[q][a] = 1
                P_X = P_XQA[:, q, a]

        if X is None:
            item_name = raw_input("What were you thinking of? - ")
            try:
                X = self.items.index(item_name)
                #add_question(item)
            except ValueError:
                self.add_item(answers, item_name)
        if X is not None:
            self.update(answers, X)

    def update(self, answers, item):
        for q in answers.keys():
            self.connections[q, :, item] *= (self.n-1.)/self.n
            self.connections[q, :, item] += answers[q]/self.n
            self.item_frequencies *= (self.n-1.)/self.n
            self.item_frequencies[item] += 1./self.n

    def add_item(self, answers, item):
        self.connections = numpy.dstack(
            (self.connections,
             numpy.zeros((len(self.questions),
                          self.max_answers,
                          1))))
        for q, answer in answers.iteritems():
            self.connections[q, :, -1] = answer
        self.item_frequencies *= (self.n-1.)/self.n
        self.item_frequencies = numpy.hstack((
            self.item_frequencies,
            numpy.array(1./self.n)))
        self.items.append(item)

    def add_question(self, question, answers):
        assert(len(answers)<=self.max_answers)
            #Extending the array is not implemented yet.
        self.connections = numpy.vstack((self.connections,
             numpy.hstack((
                 numpy.ones((1,
                             len(answers),
                             len(self.items))),
                 numpy.zeros((1,
                              self.max_answers-len(answers),
                              len(self.items)))))))
        self.questions[question]=answers
  
Yes = {"Yes.": 1., "No.": 0}
No = {"Yes.": 0., "No.": 1}
            
numbers = open20q({
    "0": {"Is X<=5?": Yes,
          "Is X>5?": No,
          "Is X odd?": No,
          "Is X even?": Yes,
          "Is X<=1?": Yes,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": No},
    "1": {"Is X<=5?": Yes,
          "Is X>5?": No,
          "Is X odd?": Yes,
          "Is X even?": No,
          "Is X<=1?": Yes,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": No},
    "2": {"Is X<=5?": Yes,
          "Is X>5?": No,
          "Is X odd?": No,
          "Is X even?": Yes,
          "Is X<=1?": No,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": No},
    "3": {"Is X<=5?": Yes,
          "Is X>5?": No,
          "Is X odd?": Yes,
          "Is X even?": No,
          "Is X<=1?": No,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": Yes},
    "4": {"Is X<=5?": Yes,
          "Is X>5?": No,
          "Is X odd?": No,
          "Is X even?": Yes,
          "Is X<=1?": No,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": No},
    "5": {"Is X<=5?": Yes,
          "Is X>5?": No,
          "Is X odd?": Yes,
          "Is X even?": No,
          "Is X<=1?": No,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": Yes},
    "6": {"Is X<=5?": No,
          "Is X>5?": Yes,
          "Is X odd?": No,
          "Is X even?": Yes,
          "Is X<=1?": No,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": No},
    "7": {"Is X<=5?": No,
          "Is X>5?": Yes,
          "Is X odd?": Yes,
          "Is X even?": No,
          "Is X<=1?": No,
          "Is X<=7?": Yes,
          "Is X close to 5?": No,
          "Is X prime?": Yes},
    "8": {"Is X<=5?": No,
          "Is X>5?": Yes,
          "Is X odd?": No,
          "Is X even?": Yes,
          "Is X<=1?": No,
          "Is X<=7?": No,
          "Is X close to 5?": No,
          "Is X prime?": No},
    "9": {"Is X<=5?": No,
          "Is X>5?": Yes,
          "Is X odd?": Yes,
          "Is X even?": No,
          "Is X<=1?": No,
          "Is X<=7?": No,
          "Is X close to 5?": No,
          "Is X prime?": No}
    })
 
numbers.identify()
