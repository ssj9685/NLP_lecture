import gensim, nltk
#nltk.download('punkt')
#print(nltk.corpus.movie_reviews.raw('neg/cv000_29416.txt')) #str
#print(nltk.corpus.movie_reviews.words('neg/cv000_29416.txt')) #list
#print(nltk.corpus.movie_reviews.sents('neg/cv000_29416.txt')) #2d list

sents = nltk.corpus.movie_reviews.sents()
model = gensim.models.Word2Vec(sents)

#cos similarity
print(model.wv.similarity('man','woman'))
print(model.wv.similarity('male','female'))
print(model.wv.similarity('pee','moon'))
print(model.wv.most_similar('sky'))