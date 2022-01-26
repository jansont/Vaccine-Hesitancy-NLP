
from sklearn.linear_model import LogisticRegression


class logistic_regression():
  '''
  logistic regression container class
  param hyperparameter_name: 'regularization' or 'max iterations' (hyperparameter being tested)
  param c: hyper parameter to test
  solver: 'saga' or 'lbfgs'  
  '''
  def __init__(self,c, hyperparameter_name, solver):
    if hyperparameter_name == 'regularization' and solver == 'saga':
      self.model = LogisticRegression(C = c, random_state=None, solver='saga', max_iter=100, multi_class = 'ovr')
    elif hyperparameter_name == 'max iteration' and solver == 'saga':
      self.model = LogisticRegression(C = 1.0, random_state=None, solver='saga', max_iter = c)
    elif hyperparameter_name == 'regularization' and solver == 'lbfgs':
      self.model = LogisticRegression(C = c, random_state=None, solver='lbfgs', max_iter = 100)
    elif hyperparameter_name == 'max iteration' and solver == 'lbfgs':
      self.model = LogisticRegression(C = 1.0, random_state=None, solver='lbfgs', max_iter = c)

  def fit(self, x_train, y_train):
    return self.model.fit(x_train, y_train)

  def predict(self,x_test):
    return self.model.predict(x_test)
