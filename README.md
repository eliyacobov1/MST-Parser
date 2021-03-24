# MST-Parser
This MST-Parser machine-learning model attempts to find the depedancy structure
of a given sentence.
For a given sentence, the model will go through the proccess of learning the correct
depedancy-tree out of all the possible depedancy trees.

It does so by iterating through all of the possible arches of the sentence and building
the Maximum-Spanning-Tree by choosing the arches that get the maximal score using a 
pre-determined feature function.
