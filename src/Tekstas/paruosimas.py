from sklearn.feature_extraction.text import CountVectorizer

documens = ['Mano vardas Margarita', 'Man 26 metai metai', 'Esu esu VCS lektore .; ))) ??']

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(documens)
#
# print(vectorizer.get_feature_names_out())
# print(X.toarray())
#
#
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
#
# words = []
# for sentence in documens:
#     for word in sentence.split():
#         words.append([word.lower()])
#
# words_encoded = encoder.fit_transform(words)
#
# print(encoder.get_feature_names_out())
# print(words_encoded.toarray())


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documens)

# TF = 1/3
# IDF = log(3/(1+1)) = log(3/2) = 0.17609125905
# Result = TF * IDF = 1/3 * 0.17609125905 = 0.05869708635

# Scikit-learn
# TF = 1 + log(1) = 1
# IDF = ln((1+N)/1+df(t)) + 1 = ln(4/2) + 1 = ln(2) + 1 = 0.69314718056 + 1 = 1.69314718056
# Result: TF * IDF = 1 * 1.69314718056 = 1.69314718056

print(tfidf.get_feature_names_out())
print(X.toarray())

from nltk.tokenize import word_tokenize
tekstas = 'hdyd lsid eoidud eoos'
word_tokenize(tekstas)