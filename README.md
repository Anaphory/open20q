# open20q
Bayes network with entropy maximiser for Twenty-Questions, Identification Key and similar games/utilities

 - This repository contains an implementation of a *naïve* 20q Bayesian classifier, and the rough outlines (including literature) of the tools, methods and technology I intend to use to build it for questions that may have dependencies, using a Tree-Augmented Naive (TAN) Bayesian classifier.

## License
All software in this repository, where not stated otherwise, should be considered under a 2-clause BSD license:

Copyright (c) 2012, 2015 Gereon Kaiping <anaphory@yahoo.de>
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
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Methodology

Following [8]: (This is for building a fixed decision tree from the probability distribution represented by the Bayesian classifier. Depending on the problem – eg. while the network is still being trained, which for an online game could be always, that's why part of the bibliography below is concerned with that part – it can be useful to not build a tree, but to use a very similar algorithm to iterate through questions.)

1. Start with the prior probabilities at the root node.

2. To choose the question for a node:

    1. For each question Q that has not been asked:

        1. Calculate the probability that the answer to the question
	     will be "yes", given the prior probabilities at that node and
        the item-question pair probabilities: the sum of 
        P(X) * P(yes | Q, X) for that question Q for all items X.

        2. Calculate the new probability distributions given a yes
        answer to that question and given a no answer to that question.

        3. Calculate the remaining bit rates of those probability
        distributions, and then multiply by the answer probabilities in
        order to come up with an expected bit rate after the question
        has been answered.

    2. Choose the question Q with the lowest expected remaining bit
    rate. Create two new nodes with the probability distributions given
    previously.

    3. Choose the questions for those two new nodes using the same
    method, unless the question was of the form "Is it X?", in which
    case the "yes" node is terminal, or unless the tree has gotten too
    deep.

The idea is to improve prior probabilities by using a Tree Augmented Naive Bayes classifier (TAN).

This has some problems:

1. Applicable information on TAN seem generally difficult to find.
2. There is no obvious implementation of TAN in python. For Naive
 Bayes, the implementation reduces to a matrix multiplication and
 similar things, as implemented in my code. For TAN it seems like a
 fully-blown Bayesian network application is necessary, and given
 [9], I would probably pick libpgm.
3. While full learning of TANs is in theory solved (although I do not
 know a good implementation for it in python), this application would
 require online-learning (because new data will be added for every
 game, and you do not want to store the full history of all games) from
 incomplete data (because in no game will players answer all possible
 questions about their chosen entity). On-line learning of TANs has
 been researched [1,2,3], so has learning from incomplete data [6], but
 last time I looked, I could not find an algorithm for doing both.

## Further Reading

[1] Alcobé, J.R., 2002. Incremental Learning of Tree Augmented Naive
Bayes Classifiers, in: Garijo, F.J., Riquelme, J.C., Toro, M. (Eds.),
Advances in Artificial Intelligence — IBERAMIA 2002, Lecture Notes in
Computer Science. Springer Berlin Heidelberg, pp. 32–41.

[2] Alcobé, J.R., 2004. Incremental Augmented Naive Bayes Classifiers,
in: Proceedings of the 16th European Conference on Artificial
Intelligence. Presented at the ECAI2004, IOS Press, Amsterdam, pp.
539–543.

[3] Alcobé, J.R., 2005. Incremental methods for Bayesian network
structure learning. AI Communications 18, 61–62.

[4] François, O.C., Leray, P. Generalized Learning Method for
the Tree Augmented Naive Bayes Classifier.

[5] Koller, D., Friedman, N., 2009. Probabilistic graphical models:
principles and techniques. MIT press.

[6] Leray, P., François, O., 2005. Bayesian Network Structural
Learning and Incomplete Data [WWW Document]. URL
http://eprints.pascal-network.org/archive/00001232/ (accessed
10.21.14).

[7] Webb, G.I., Boughton, J.R., Wang, Z., 2005. Not So Naive Bayes:
Aggregating One-Dependence Estimators. Mach Learn 58, 5–24.
doi:10.1007/s10994-005-4258-6

[8] Sitaker, K.J, 2010. Bayesian 20 Questions. Kragen thinking out
loud – half-baked ideas 3/2010.
http://lists.canonical.org/pipermail/kragen-tol/2010-March/000912.html,
still available on
http://web.archive.org/web/20130125014055/http://lists.canonical.org/pipermail/kragen-tol/2010-March/000912.html

[9] http://stackoverflow.com/questions/14916351/learning-and-using-augmented-bayes-classifiers-in-python
