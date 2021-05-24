# Markov-Model
## JAVA CODE IMPLEMENTATION
1.	Please refer to java script provided in the Zip.

(a)	The script “HiddenMarkov.java” implements Likelihood and Decoding problem in Hidden Markov Model.
(b)	Follow on-screen instruction to run the script on any case that can be modelled as HMM.
(c)	The likelihood problem is solved using forward algorithm and decoding problem is solved using Viterbi algorithm as discussed in this report.

2.	The script “LearningProb.java” implements Learning problem of Hidden Markov Model.
(a)	To successfully run the training model please provide the array of sequence of observation in variable xs at line 306.
(b)	Modify line 286 for states and 289 for emission symbols. It is assumed that emission state is represented by a single character between ‘0’ to ‘1’ and/or ‘A’ to ‘Z’ (capital letters only). Thus at max 36 emission states can be provided.
(c)	The emission state symbols can simply be given as a single array.
(d)	The each sequence of emission state observation in xs array is given as a single string for faster computation. 
(e)	Since there are a lot of precision errors when dealing with large decimals. The Log probability is used instead of traditional multiplicative probability calculations in implementation of Baum-Welch Algorithm.
