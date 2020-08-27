import konlpy

#print(konlpy.corpus.kolaw.fileids())
#print(konlpy.corpus.kobill.fileids())

#f = konlpy.corpus.kolaw.open('constitution.txt')
#print(f)
#print(f.read())
#f.close()

kolaw = konlpy.corpus.kobill.open('1809890.txt').read()

#pos : part of speech
pos = konlpy.tag.Hannanum().pos(kolaw[:1000])
print(pos[:5])