# According to the ideas of http://lists.canonical.org/pipermail/kragen-tol/2010-March/000912.html

import numpy
from collections import OrderedDict

class open20q (object):
   additional = ["I don't know."]
   final = "Are you thinking of %s?"
   YN = {"Yes.": 1., "No.": 0}

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
      for item in self.items:
         items[item].setdefault(self.final%item, YN)
      for questions in items.values():
         for (question, answers) in questions.iteritems():
            known_answers = self.questions.setdefault(question, additional[:])
            for answer in answers:
               if answer not in known_answers:
                  known_answers.append(answer)
            max_answers = max(max_answers, len(known_answers))
      self.connections = numpy.zeros((len(items), len(self.questions), max_answers))
      for i, item in enumerate(self.items):
         for q, (question, answers) in enumerate(self.questions.iteritems()):
            for a in xrange(len(answers)):
               self.connections[q, a, i] = items.get(item, {}).get(question, {}).get(answers[a], 0)
      
            
items = [
    "1","2","3","4","5","6","7","8","9","0"
    ]
a_priori_Ps = numpy.array([
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    ])
questions = ["Is it %s?"%item for item in items] + [
    "X<=5 ",
    "X>5 ",
    "X odd ",
    "X even ",
    "X<=1 ",
    "X<=7 ",
    "X close to 5 ",
    "X is prime ",
    ]
connections = numpy.vstack((
    numpy.eye(len(items)),
    numpy.array(
    [[1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
     [0,.1,.8, 1,.9, 1,.8,.1, 0, 0],
     [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
     ])))

C_QX = connections
n=50

def xlogx(x):
   x = numpy.asarray(x)
   l = numpy.zeros_like(x)
   l[x>0] = numpy.log(x[x>0])
   return x * l

def rbr(Ps):
    # remaining bit rate
    Ps = numpy.asarray(Ps)
    Ps = Ps / sum(Ps)
    Ps[numpy.isnan(Ps)] = 0
    return -sum(xlogx(Ps))/numpy.log(2)

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

def P_yes(P_X):
    return numpy.dot(C_QX, P_X)

def rbr_after_answer(P_Q, P_XQ, P_XnotQ):
    rbrs = P_Q * rbr(P_XQ) + (1-P_Q) * rbr(P_XnotQ)
    rbrs[P_Q == 0] = numpy.inf 
    rbrs[P_Q == 1] = numpy.inf 
    return rbrs 

def update(answers, item):
    global C_QX
    global a_priori_Ps
    global n
    C_QX[answers!=0.5][:, item] = (n-1.)/n*C_QX[answers!=0.5][:, item] + answers[answers!=0.5]/n
    a_priori_Ps *= (n-1)/float(n)
    a_priori_Ps[item] += 1./n

def add_item(answers, item):
    global C_QX
    global items
    global a_priori_Ps
    i = len(items)
    X = numpy.hstack((1,
                      numpy.zeros(i),
                      answers[i:]))
    X.shape = (-1, 1)
    C_QX = numpy.hstack((X, numpy.vstack((numpy.zeros(i), C_QX))))
    items.insert(0, item)
    questions.insert(0, ("Is it %s?" % item))
    a_priori_Ps = numpy.hstack((1./n, (n-1.)/n*a_priori_Ps))

def add_question(item):
    global C_QX
    global questions
    question = raw_input("??")
    questions.append(question)
    answers = numpy.ones(len(items))*0.5
    answers[item] = 1
    C_QX = numpy.vstack((C_QX, answers))
    
Xmax = 10
epsilon = 0.1
def guess():
        global Xmax
        R = numpy.random.randint(Xmax+1)
        if Xmax==R:
            Xmax+=1
        print "R:",R 
        P_X = a_priori_Ps/sum(a_priori_Ps)
        answers = numpy.ones(len(questions))*0.5
        counter = 0
        while 1:
            for i in xrange(len(items)):
                print items[i], P_X[i],
            print
            P_Q = P_yes(P_X)
            P_XQ = bayes(C_QX, P_X, P_Q)
            P_XnotQ = bayes(1-C_QX, P_X, 1-P_Q)
            rbrs = rbr_after_answer(P_Q, P_XQ, P_XnotQ)
            rbrs[answers!=0.5]=numpy.inf
            print rbr(P_X)
            for q in xrange(len(questions)):
                print rbrs[q], questions[q]
            question = rbrs.argmin()
            if numpy.random.random()<epsilon:
                print "Random Question:"
                question = numpy.random.randint(len(items), len(questions))
            else:
                print "Best Question:"
            answer = raw_input(questions[question])
            counter += 1
            if answer.startswith("n") or answer.startswith("N"):
                answers[question] = 0
                P_X = P_XnotQ[:, question]
                if counter > 4:
                    item = -1
                    break
            else:
                answers[question] = 1
                P_X = P_XQ[:, question]
                if question < len(items):
                    item = question
                    break
                if counter > 4:
                    item = -1
                    break

        if item == -1:
            item_name = raw_input()
            try:
                item = items.index(item_name)
                #add_question(item)
            except ValueError:
                P_Q[answers!=0.5] = answers[answers!=0.5]
                add_item(P_Q, item_name)
        if item != -1:
            update(answers, item)
        print "Yay."

while 1:
   guess()
