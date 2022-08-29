# ICL-Final-Project
This is the code used to run experiments for my final project Learning with Guarantees. Three approaches are compared: SA compression, PAC-Bayes, and test set bounds.
The code used to compute the bound in SA Compression is based on MATLAB code from https://marco-campi.unibs.it/pdf-pszip/Wait-and-judge.PDF, using a bisectional numerical algorithm to solve the equation required for the bound.

We build on top of existing code in implementing the PAC-Bayesian and test set methods. For PAC-Bayes we most heavily rely on the existing code at https://github.com/mperezortiz/PBB. Our most important modifications to this code base are 

1) incorporating the code enabling optimization of inverted binary kl from https://github.com/eclerico/CondGauss

2) incorporating code enabling optimization through the CondGauss approach from https://github.com/eclerico/CondGauss

3) building in the ability to run training with the averaged posterior approach of https://arxiv.org/pdf/1905.13367.pdf

For the test set bounds we adapt implementations from https://github.com/cambridge-mlg/pac-bayes-tightness-small-data.
All experiments can be run by adjusting arguments in the runexp.py files of each approach.
