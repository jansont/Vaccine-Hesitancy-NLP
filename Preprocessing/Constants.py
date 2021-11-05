'''
  Author: Theodore Janson <theodore.janson@mail.mcgill.ca>
  Source Repository: https://github.com/jansont/VaccinationAnalysis
'''
import nltk
from nltk.corpus import stopwords
# clearnltk.download('stopwords')


pdict = {1:'to_lower', 2: 'handle_emojis', 3: 'remove_numbers', 4:'remove_punctuation', 
            5:'tokenise', 6:'remove_tags', 7:'remove_urls', 8:'remove_stopwords', 
            9: 'replace_abbreviations', 10: 'lemmatize', 11: 'stem', 12: 'detokenise'}

PIPELINE = [pdict[i] for i in [2,1,3,4,5,6,7,9,8,10,12]]

#punctuation to remove from text
PUNCTUATION = ['.',',',':',';','-','/','@','#','%','^','&','*','(',')','_','=','+','~','`',"'",'"']

#abbreviations to expand 
ABBREVIATIONS = {'$': 'dollars', 'btw': 'by the way', 'lol':'laugh', 'lmao': 'laugh', 
                 'wtf': 'fuck', 'lmao': 'laugh', 'tmrw': 'tomorrow', 'tbh': 'honest', 'rofl': 'laugh',
                 'lmk': 'know', 'nvm': 'nevermind', 'imo': 'opinion', 'idk': 'do not know' }


#word to remove fromt text as they don't convey much useful information
#retain negatives as they are useful for sentiment classification
STOPWORDS = [word for word in stopwords.words("english") if word not in ['no', 'not','nor']]

#filter out tweets containing these terms 
TOPICS = ['vaccin', 'pfizer', 'moderna', 'astrazeneca', 'biontech', 'oxfordvaccine', 'vaccineswork', 'vaccinework', 'azvaccine',
'covidiots', 'endthelockdown', 'greatreset', 'plandemic', 'mrna', 'eugenics', 'thisisourshot','kungflu', 'rna', 'gavi', 'depopulation', 'peoplesbodyyourchoice',
'iwillnotcomply', 'mybodymychoice', 'glyphosate', 'vax', 'cepi', 'nvicm', 'mercury', 'pharm', 'testing', 'gates', 
'toxic', 'herd', 'sheeple', 'johnson', 'oxford', 'zycov', 'janssen', 'sputnik', 'convidecia' , 'coronavac', 'medigen',
'soberana', 'abdala', 'allergy','anaphylaxis', 'antibody','antiviral', 'booster', 'immun', 'conspir', 'theor', 'passport', 'jab', ]


