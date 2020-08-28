# Day_03_01_Doc2Vec.py
import nltk
import random
import collections
import time


def make_vocab(vocab_size=2000):
    #nltk.download('movie_reviews')

    words = nltk.corpus.movie_reviews.words()
    print(words)            # ['plot', ':', 'two', 'teen', 'couples', 'go', 'to', ...]
    print(len(words))       # 1583820

    # 문제
    # 불용어와 추가적으로 구두점 등을 삭제해서 2000개를 만드세요
    # all_words = nltk.FreqDist([w.lower() for w in words])
    all_words = collections.Counter([w.lower() for w in words])
    most_2000 = all_words.most_common(vocab_size)
    print(most_2000[:5])

    return [w for w, _ in most_2000]


# 문제
# 영화 리뷰를 피처로 변환하는 함수를 만드세요
def make_feature(doc_words, vocab):
    feature, uniques = {}, set(doc_words)
    for v in vocab:
        feature['has_{}'.format(v)] = (v in uniques)

    return feature


vocab = make_vocab()

# print(nltk.corpus.movie_reviews.fileids())
# ['neg/cv000_29416.txt', 'neg/cv001_19502.txt', ...]

doc = nltk.corpus.movie_reviews.words('neg/cv000_29416.txt')
#print(doc)
# print(len(doc))     # 879
# ['plot', ':', 'two', 'teen', 'couples', 'go', 'to', ...]

print(make_feature(doc, vocab))

print(nltk.corpus.movie_reviews.categories())   # ['neg', 'pos']

neg = nltk.corpus.movie_reviews.fileids('neg')
pos = nltk.corpus.movie_reviews.fileids('pos')
print(len(neg), len(pos))                     # 1000 1000

neg_data = [nltk.corpus.movie_reviews.words(filename) for filename in neg]

# 문제
# 80%로 학습하고 20%에 대해 예측하세요
start = time.time()
neg_data = [(make_feature(nltk.corpus.movie_reviews.words(filename), vocab), 'neg') for filename in neg]
pos_data = [(make_feature(nltk.corpus.movie_reviews.words(filename), vocab), 'pos') for filename in pos]
print("spend time : ",time.time()-start)

#unbalanced!!
#data = neg_data + pos_data
#random.shuffle(data)
#train_set, test_set = data[400:], data[:400]        # 1600, 400

random.shuffle(neg_data)
random.shuffle(pos_data)

train_set = neg_data[:800]+pos_data[:800]
test_set = neg_data[800:]+pos_data[800:]

clf = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(clf, test_set))




