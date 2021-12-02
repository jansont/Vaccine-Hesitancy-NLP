'''
  Author: Theodore Janson <theodore.janson@mail.mcgill.ca>
  Source Repository: https://github.com/jansont/VaccinationAnalysis
'''
import numpy as np

class MultinomialNaiveBayes():
  def __init__(self, alpha=1,hyperparameter_name=None,solver=None ):
    '''
    param alpha: constant for Laplace smoothing
    param hyperparameter_name: None. Allows for the class to be instantiated in the model tester class. 
    param hyperparameter_name: None. Allows for the class to be instantiated in the model tester class. 
    '''
    self.alpha= alpha
    return

  def fit(self,x,y):
     '''
     param x: train set features
     param y: train set labels
     '''
     N, D = x.shape 
     C = np.max(y) + 1 #find the number of classes 
     alpha=self.alpha #get the hyper parameter alpha 
     #theta represents the conditional probiliy of each class given each feature  
     theta= np.zeros((C,D))
     Nc = np.zeros(C) # number of instances in class c 
     # for each class get the conditional probility of the class given feature
     for c in range(C):
            x_c = x[y == c]                           #slice all the elements from class c
            Nc[c] = x_c.shape[0]                      #get number of elements of that class 
            theta[c,:] = np.sum(x_c,0)                  #sum of features of class c (find the numerator)
    #theta is set to its proper value as the conditional probility of the class given the feature 
     theta[:]=(theta[:]+alpha)/(np.sum(x,0)+D+1) #divide by total instances of the words while also doing laplace smooting 
     self.theta= theta
     self.pi= (Nc+1)/(N+C) #smoothing for the prior 
     return self

  def predict(self, xt):
      '''
      param xt: test set features
      returns: prediction for labels on test set
      '''
      Nt, D = xt.shape
      log_prior = np.log(self.pi)[:, None] # calculate the log of the prior
      # find the log sum of the input multiplied by theta elementwise for each class and then add the log of the prior of that class
      #this gives us values proportional to the relative probabilities of each class
      def find_prob_of_classes(sparse): # calculate the probability of each calss given the inputs xt 
        sample_likelihood = [] 
        for c in range(self.theta.shape[0]):
          likelihood = sparse*np.log(self.theta[c])
          likelihood = np.sum(likelihood) #+log_prior[c][0] ##why do we do this twice ?? we seem to be multiplying by the prior twice what is this period too?
          likelihood = likelihood + log_prior[c][0]
          sample_likelihood.append(likelihood)
        return sample_likelihood
      y_prob = [] 
      for element in xt:
        y_prob.append(find_prob_of_classes(element))
      y_pred = np.argmax(y_prob,1)#return most likely class for each input not probability

      return y_pred

