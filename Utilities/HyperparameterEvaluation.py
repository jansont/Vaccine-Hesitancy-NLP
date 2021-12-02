class sets():
  '''Container for the dataset being tested'''

  def __init__(self, X_train, Y_train, X_test, Y_test, test_prop = 0.2, L = 5, rel_size = 1.0, max_test_size = 1000):
    '''
    param X_train: training set features
    param Y_train: labels for training set
    param X_test: test set features 
    param Y_test: labels for test set
    param test_prop: size of test set relative to train set (default 0.2: test set is 1/5 the size of the training set)
    param L: number of folds for L fold cross validation
    param rel_size: size of desired training set relative to available data
    param max_test_set_size: max size of test set (to speed up computation)
    '''
    train_set_size = X_train.shape[0] 
    X_train = X_train[:int(train_set_size*rel_size)] #slice desired train set size
    Y_train = Y_train[:int(train_set_size*rel_size)]
    test_set_size = int(train_set_size * test_prop) #get appropriate number of test examples 
    if test_set_size > max_test_size: 
      test_set_size = max_test_size #cap off the size of the test set if necessary

    self.x_train = X_train
    self.y_train = np.array(Y_train)
    self.x_test = X_test[:test_set_size]
    self.y_test = np.array(Y_test[:test_set_size]) #slice test set
    self.L = L

  def cross_validation_split(self):
    '''
    L fold cross validation function which generates subsets of the training set
    yield x_train: subtraining set features
    yield y_train: subtraining set labels
    yield x_val: validation set features
    yield y_val: validation set labels
    '''
    dataset_size = self.y_train.shape[0]
    indices = list(range(dataset_size))
    num_val = dataset_size // self.L #size of validation set
    for l in range(self.L):
      validation_indices = list(range(l * num_val, (l+1)*num_val)) #get indices for the next validation set 
      train_indices = [x for x in indices if x not in validation_indices]  #get indices for the sub train set
      x_train, x_val = self.x_train[train_indices], self.x_train[validation_indices] #slice out
      y_train, y_val = self.y_train[train_indices], self.y_train[validation_indices]
      yield x_train, y_train, x_val, y_val
      

class model_tester():
  '''Class used to test the models using cross validation'''

  def __init__(self, model, datasets, hyperparameters, hyperparameter_name = 'regularization', solver = 'saga', val_limit = 500):
    '''
    param model: class variable of the model being tested 
    param hyperparameters: array of hyperparameters being tested
    param hyperparameter_name: string of hyperparameter name for displaying results
    param solver: solver type for logistic regression (set to None for Naive Bayes)
    param val_limit: limit on the number of example used for validation testing (to speed things up)
    '''
    self.model_class = model
    self.datasets = datasets
    self.hyperparameters = hyperparameters
    self.hyperparameter_name = hyperparameter_name
    self.solver = solver
    self.val_limit = val_limit

  def initialize_model(self, h):
    '''Initializing the model with the desired hyperparameter h'''
    return self.model_class(h, self.hyperparameter_name, self.solver)

  def test(self):
    '''Tester used for cross validation
    returns metrics: object containing all the performance metrics'''
    num_class = np.unique(self.datasets.y_train).shape[0] #get the number of classes (used for metrics)
    metrics = model_metrics(self, self.datasets) #initialize our metrics object
    for i,h in enumerate(self.hyperparameters): #iterate over the hyperparameters
      for f, sets in enumerate(self.datasets.cross_validation_split()): #for each hyperparameter, iterate over the validation sets
        (x_train, y_train, x_val, y_val) = sets #get the validation set, and the sub trainset
        x_val, y_val = x_val[:self.val_limit], y_val[:self.val_limit] #slice out to the desired max size (speed things up)  
        model = self.initialize_model(h)         #initialize the model
        start = time.time()
        y_val_pred = model.fit(x_train, y_train).predict(x_val) #fit training data and predict on validation data
        timing = time.time() - start #get the time it took to train and predict
        print(f'...Training and testing for hyperparameter {i+1} on validation fold {f+1}') 
        metrics.update_metrics(True, i, f, y_val_pred, y_val, timing) #update the metrics
      model = self.initialize_model(h) #initialize the model again for the test set
      start = time.time()
      y_prediction = model.fit(self.datasets.x_train, self.datasets.y_train).predict(self.datasets.x_test) #fit training data and predict on test data     
      timing = time.time() - start
      print(f'...Training and testing for hyperparameter {i+1} on test set')
      metrics.update_metrics(False, i, f, y_prediction, self.datasets.y_test, timing) #update metrics again
    return metrics




class model_metrics(): 
# '''This class is used to hold the performance metrics on all the validation tests and tests on the test set'''
  def __init__(self, tester, datasets):
    '''
    param tester: tester object (used to pass the hyperparameter array and hyperparameter name)
    datasets: sets object containing validation count L
    '''
    L  = datasets.L #Validation count
    self.hyperparameter_name = tester.hyperparameter_name 
    self.hyperparameters = tester.hyperparameters
    hyperparameter_range = tester.hyperparameters.shape[0]
    #initialize the arrays containing the performance metrics
    self.accuracy = [np.zeros((hyperparameter_range, L)), np.zeros((hyperparameter_range))] #accuracy 
    self.recall = [np.zeros((hyperparameter_range, L)), np.zeros((hyperparameter_range))] #recall
    self.precision = [np.zeros((hyperparameter_range, L)), np.zeros((hyperparameter_range))] #precision
    self.f1 = [np.zeros((hyperparameter_range, L)), np.zeros((hyperparameter_range))] #f1
    self.timing = [np.zeros((hyperparameter_range, L)), np.zeros((hyperparameter_range))] #time to train and test
  
  def show_results(self, log = None):
    '''Display plots of performance vs hyperparameter
    param log: if True is passed, scaled of x axis is log scale 
    '''
    plt.suptitle('Peformance Metrics on Validation and Test Sets',fontsize=20, x =0.4)
    m = {'Accuracy': self.accuracy, 'Recall':self.recall, 'Precision': self.precision, 'F1': self.f1, 'Timing':self.timing}
    fig = plt.figure(figsize= (20,12))
    fig.add_subplot(2,1,1)
    for i,metric in enumerate(m):
      if metric != 'Timing':
        plt.plot(self.hyperparameters, m[metric][1],  label=(f'{metric} on test set'))
        plt.errorbar(self.hyperparameters, np.mean(m[metric][0], axis=1), np.std(m[metric][0], axis=1), label=(f'{metric} on validation set'))
        plt.legend(loc = 'lower right'), plt.xlabel(self.hyperparameter_name), plt.ylabel(metric), plt.grid(True)
        if log != None:
          plt.xscale("log")
      fig.tight_layout(rect=[0.03, 0.03, 0.8, 0.97])
    else: 
        fig.add_subplot(2,1,2)
        plt.plot(self.hyperparameters, m[metric][1],  label=(f'{metric} on test set'))
        plt.errorbar(self.hyperparameters, np.mean(m[metric][0], axis=1), np.std(m[metric][0], axis=1), label=(f'{metric} on validation set'))
        plt.legend(), plt.xlabel(self.hyperparameter_name), plt.ylabel(metric), plt.grid(True) 
        if log != None:
          plt.xscale("log")
        fig.tight_layout(rect=[0.03, 0.03, 0.8, 0.97])   
    plt.show()
  
  def update_metrics(self,validation, i, f, y_pred, y_test, timing): 
    '''
    Updating the arrays to contain the performance on the latest test
    param validation: boolean (True for validation test, False for test set test)
    param y_pred: array of prediction
    param y_test: array of true labels
    param timing: time for latest train and predict
    '''
    num_class = np.unique(y_test).shape[0] 
    if num_class < 3:
      avg ='binary' 
    else: 
      avg = 'micro' #multiclass precision, recall and f1 requires the parameter micro
    if validation == True: 
      self.accuracy[0][i, f] = self.evaluate_acc(y_pred, y_test.flatten()) #get accuracy and update
      self.precision[0][i, f] = self.evaluate_precision(y_pred, y_test.flatten(), avg)  #get precision and update
      self.recall[0][i, f] = self.evaluate_recall(y_pred, y_test.flatten(), avg)  #get recall and update
      self.f1[0][i, f] = self.evaluate_f1(y_pred, y_test.flatten(), avg)  #get f1 and update
      self.timing[0][i,f] = timing #update timing
    else: 
      self.accuracy[1][i] = self.evaluate_acc(y_pred, y_test.flatten())
      self.precision[1][i] = self.evaluate_precision(y_pred, y_test.flatten(), avg)
      self.recall[1][i] = self.evaluate_recall(y_pred, y_test.flatten(), avg)
      self.f1[1][i] = self.evaluate_f1(y_pred, y_test.flatten(), avg)
      self.timing[1][i] = timing

  def get_best(self):
    '''
    Returns the best hyperparameter on the validation sets using the best average accuracy
    Returns the best average accuracy on validation sets 
    Returns the accuracy of the model: test set accuracy for the hyperparameter at which validation had the best average accuracy
    Returns the best hyperparameter on the test set using ccuracy
    Returns the best average accuracy on test set
    '''
    best_accuracy_val = np.mean(self.accuracy[0], axis=1).max()
    best_hyperparameter_val = self.hyperparameters[np.argmax(np.mean(self.accuracy[0], axis=1))]
    best_accuracy_t = self.accuracy[1].max()
    best_hyperparameter_t = self.hyperparameters[np.argmax(self.accuracy[1])]
    i = np.where(self.hyperparameters == best_hyperparameter_val)[0][0]
    accuracy = self.accuracy[1][i]
    print(f'\nHighest average accuracy on validation set: {best_accuracy_val :.3f} with {self.hyperparameter_name} of {best_hyperparameter_val}')
    print(f'\nAccuracy on test set:  {best_accuracy_val :.3f} with {self.hyperparameter_name} of {best_hyperparameter_val}')
    print(f'\nAccuracy:  {accuracy :.3f}')
    return best_accuracy_val, best_hyperparameter_val, accuracy, best_accuracy_t, best_hyperparameter_t

  def evaluate_acc(self,y_prediction, y_test): #return accuracy using prediction and labels
    return np.sum(y_prediction == y_test) / y_test.shape[0]

  def evaluate_recall(self,y_prediction, y_test, avg): #return recall using prediction and labels
    return recall_score(y_test, y_prediction, average = avg)
    
  def evaluate_precision(self,y_prediction, y_test, avg): #return precision using prediction and labels
    return precision_score(y_test, y_prediction, average = avg)

  def evaluate_f1(self,y_prediction, y_test, avg): #return f1 using prediction and labels
    return f1_score(y_test, y_prediction, average = avg)

  def split_dataset(df, training_prop = 0.8, test_prop = 0.5):
  training, test_validation = train_test_split(df, 
                                      test_size= ( 1 - training_prop ), 
                                      stratify=df['Score'].values)

  validation, test = train_test_split(test_validation, 
                                   test_size=test_prop, 
                                   stratify=test_validation['Score'].values)
  return training, validation, test


  class vectorizer():
  '''Vectorizer class
  param vect: vectorizer class
  param n_grams: tuple of the number of grams
  param stopwords: we used the stopwords from NLTK
  param strip_accents: boolean
  param lowercase: boolean
  '''
  def __init__(self, vect, n_grams, stopwords, strip_accents = None, lowercase = True):
    self.stopwords = stopwords #list or str'english'
    self.n_grams = n_grams #tuple (range)
    self.strip_accents = strip_accents #'ascii or None
    self.lowercase = lowercase
    self.vector = vect(strip_accents = self.strip_accents, lowercase = self.lowercase, stop_words = stopwords, ngram_range= n_grams)

  def fit_transform(self, corpus):
    return self.vector.fit_transform(corpus)

  def transform(self, text):
    return self.vector.transform(text)
