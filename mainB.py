import nltk
import numpy as np
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# citirea datelor
data_path = '.'
train_data_df = pd.read_csv('train_data.csv')
test_data_df = pd.read_csv('test_data.csv')

# codificam etichetele / labels in valori cu numere intregi dela 0 la N
etichete_unice = train_data_df['label'].unique()
label2id = {}
id2label = {}
for idx, eticheta in enumerate(etichete_unice):
    label2id[eticheta] = idx
    id2label[idx] = eticheta


#Exercitiu: aplicati dictionarul label2id peste toate etichetele din train
labels = []

for eticheta in train_data_df['label']:
    labels.append(label2id[eticheta])
labels = np.array(labels)

stop_words_german = set(stopwords.words('german'))
stop_words_dutch = set(stopwords.words('dutch'))
stop_words_danish = set(stopwords.words('danish'))
stop_words_spanish = set(stopwords.words('spanish'))
stop_words_italian = set(stopwords.words('italian'))


def proceseaza(text):
    # text = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
    # text = text.replace('\n', ' ').strip().lower()
    # text_in_cuvinte = text.split(' ')
    # return text_in_cuvinte
    text = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
    text = text.replace('\n', ' ').strip().lower()
    text_in_cuvinte = text.split(' ')
    words_without_stop_words = []
    for word in text_in_cuvinte:
        if word not in (stop_words_german and stop_words_dutch and stop_words_danish
                        and stop_words_spanish and stop_words_italian):
            words_without_stop_words.append(word)
    return set(words_without_stop_words)

#aplicam functia de procesare pe datele de antrenare
data = train_data_df['text'].apply(proceseaza)


# putem imparti datele de antrenare astfel:
# 20% date de test din total
# 15% date de validare din ce ramane dupa ce scoatem datele de antreanre

nr_test = int(20/100 * len(train_data_df))
print("Nr de date de test: ", nr_test)

nr_ramase = len(data) - nr_test
nr_valid = int(15/100 * nr_ramase)
print("Nr de date de validare: ", nr_valid)

nr_train = nr_ramase - nr_valid
print("Nr de date de antrenare: ", nr_train)

#amestecam datele pentru ca exista sanse ca datele sa fie intr-o ordine care nu reflecta realitatea. adica
#se poate ca toate sa fie ordonate dupa eticheta si atunci acestea s ar antrena pe cate o limba pe rand
# luam niste indici de la 0 la N
indici = np.arange(0,len(train_data_df))
# ii permutam si apoi putem sa-i folosim pentru a amesteca datele
np.random.shuffle(indici)


# facem impartirea in ordinea in care apar datele
# datele se amesteca folosind indicii permutati, in loc de split in functie
# ordinea in care apar exemplele
train_data = data[indici[:nr_train]]
train_labels = labels[indici[:nr_train]]

valid_data = data[indici[nr_train : nr_train + nr_valid]]
valid_labels = labels[indici[nr_train : nr_train + nr_valid]]

test_data = data[indici[nr_train + nr_valid: ]]
test_labels = labels[indici[nr_train + nr_valid:]]


# folosim CountVect pt a transofmra textul intr-un vector bazat pe frecventa cuvintelor
#The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide,
# returning a "best fit" hyperplane that divides your data.
vectorizer = CountVectorizer(tokenizer = lambda x:x,    # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor = lambda x:x,  #  data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features = 37000) #cred ca sunt primele 37000 de cuvinte cele mai intalnite
vectorizer.fit(train_data)
X_train = vectorizer.transform(train_data)#transforma documentul in matrice
X_valid = vectorizer.transform(valid_data)
X_test = vectorizer.transform(test_data)


model = svm.LinearSVC(C=0.001)
#cu 1000 features--------60-0.001, 62 - 0.01, 60 - 0.1, 60 - 0.5, 54-1, 59 - 10, 47-100, 53- 1000, 61-0.2,
#cu 35000----69-0.01, 70-0.001, 61-0.0001, 66-0.1, 65-0.5, 65-0.2,
model.fit(X_train, train_labels)#potrivim train_labels in X_train(labels in features)
vpreds = model.predict(X_valid)
tpreds = model.predict(X_test)


print('Acuratete pe validare ', accuracy_score(valid_labels, vpreds))#vedem acuratetea
print('Acuratete pe test ', accuracy_score(test_labels, tpreds))


def matrice_confuzie(x,y):#facem matricea de confuzie
    labels = unique_labels(test_labels)
    column = [f'Predicted {label}' for label in labels]
    indices = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(x,y), columns=column, index=indices)
    print(table)

matrice_confuzie(test_labels, tpreds)

# #citim datele de test pentru kaggle
# test_data_df = pd.read_csv('test_data.csv')
# # preprocesam datele (impartim in tokens)
# date_test_procesate = test_data_df['text'].apply(proceseaza)
#
# # aplicam metoda BoW de vectorizare pe datele pre-procesate
# date_test_vectorizate = vectorizer.transform(date_test_procesate)
# # obtinem predictii
# predictii = model.predict(date_test_vectorizate)
#
# predictii2 = []
# for i in predictii:
#     if i == 0:
#         predictii2.append('Ireland')
#     elif i == 1:
#         predictii2.append('England')
#     elif i == 2:
#         predictii2.append('Scotland')
#
# rezultat = pd.DataFrame({'id': np.arange(1, len(predictii2)+1), 'label': predictii2})
# # putem numi fisierul in functie de hiperparametrii si model, ca sa nu uitam ce am incercat
# nume_model = str(model)
# nr_de_caracteristici = 'N_35000_c0001'
# functie_preprocesare = 'vectorizare'
# nume_fisier = '_'.join([nume_model, nr_de_caracteristici, functie_preprocesare]) + '.csv'
# # salvam rezultatul fara index in fisierul submission.csv
# # e obligatoriu sa-l numim submission.csv altfel nu stie kaggle care output e rezultatul
# rezultat.to_csv(nume_fisier, index=False)


# kf = KFold(n_splits=5)
# toate_datele_vectorizate = vectorizer.transform(data)
# # print(toate_datele_vectorizate.shape)
# toate_etichetele = train_data_df['label'].values
# # print(kf.get_n_splits(toate_datele_vectorizate,toate_etichetele))
# def acuratete_antrenare():
#     i = 0
#     acuratete_totala = 0
#     for train_index, test_index in kf.split(toate_datele_vectorizate, toate_etichetele):
#         #print(train_index, test_index)
#         X_train_cv, X_test_cv = toate_datele_vectorizate[train_index], toate_datele_vectorizate[test_index]
#         y_train_cv, y_test_cv = toate_etichetele[train_index], toate_etichetele[test_index]
#         #print(X_train_cv.shape)
#         model = svm.LinearSVC(C=0.5)
#         model.fit(X_train_cv, y_train_cv)
#         tpreds = model.predict(X_test_cv)
#         print(accuracy_score(y_test_cv, tpreds))
#         acuratete_totala+= accuracy_score(y_test_cv, tpreds)
#         i+=1
#     print('Acuratetea medie este %.5f' %(acuratete_totala/i))
#
# acuratete_antrenare()