#gensim
import gensim, konlpy, nltk

def show_doc2vec():
    f = open('data/computer.txt','r',encoding='utf-8')
    stop_words = ['a','the','of','and','for','to']
    words_list = [[word for word in line.lower().strip().split() if word not in stop_words] for line in f]
    f.close()

    freq = nltk.FreqDist(w for words in words_list for w in words)
    words_mul = [[word for word in words if freq[word] >1 ] for words in words_list]

    dct = gensim.corpora.Dictionary(words_mul)
    #bag of words => bow
    vectors = [dct.doc2bow(words) for words in words_mul]
    print(dct)
    print(dct.token2id)
    print(vectors)

def show_word2vec():
    def save():
        text = ['I love you','I hate you']
        token = [s.split() for s in text]
        #print(token)

        embedding = gensim.models.Word2Vec(token, min_count=1, size=5, sg=True)
        #print(embedding)
        #print(embedding.wv['I'])
        #print(embedding.wv['you'])

        #print(embedding.wv.vectors)
        embedding.save('data/word2vec.out')
    save()
    embedding = gensim.models.Word2Vec.load('data/word2vec.out')
    print(embedding.wv['I'])

show_word2vec()


#print(words_list)
#print(words_mul)
