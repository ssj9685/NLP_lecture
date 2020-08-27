# Day_04_04_review.py

#Q
#gutenberg

import nltk
#nltk.download('gutenberg')
#print(nltk.corpus.gutenberg.fileids())
from nltk.corpus import gutenberg as gtb

hamlet = nltk.Text(gtb.words('shakespeare-hamlet.txt'))
print(hamlet.concordance('gertrude'))