'''
  Author: Theodore Janson <theodore.janson@mail.mcgill.ca>
  Source Repository: https://github.com/jansont/VaccinationAnalysis
'''
from sklearn.model_selection import GridSearchCV


class Grid_Search():
  def __init__(self, model, search_parameters):
    self.search_parameters = search_parameters
    self.model = model
    self.searcher = GridSearchCV(model, search_parameters)

  def fit(self, X=None):
    self.searcher.fit(X=None)
    return self.searcher
  
  def get_best(self):
    return self.searcher.best_estimator_, self.searcher.best_params_

  def get_best_scores(self, features):
    return self.searcher.best_estimator_.perplexity(features), self.searcher.best_scores_