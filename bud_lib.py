import pandas as pd
import numpy as np
from spacy.lang.pl import Polish
from tqdm.notebook import tqdm
import morfeusz2
morf = morfeusz2.Morfeusz()#generate=False)
import re
nlp = Polish()

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoderLabse = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")

#custom stopwords
stopwords = ['i_tak_dalej','i_tym_podobne','wraz','a','aby','ach','acz','aczkolwiek','aj','albo','ale','alez','ależ','ani','az','aż','bardziej','bardzo','beda','bedzie','bez','deda','będą','bede','będę','będzie','bo','bowiem','by','byc','być','byl','byla','byli','bylo','byly','był','była','było','były','bynajmniej','cala','cali','caly','cała','cały','ci','cie','ciebie','cię','co','cokolwiek','cos','coś','czasami','czasem','czemu','czy','czyli','daleko','dla','dlaczego','dlatego','do','dobrze','dokad','dokąd','dosc','dość','duzo','dużo','dwa','dwaj','dwie','dwoje','dzis','dzisiaj','dziś','gdy','gdyby','gdyz','gdyż','gdzie','gdziekolwiek','gdzies','gdzieś','go','i','ich','ile','im','inna','inne','inny','innych','iz','iż','ja','jak','jakas','jakaś','jakby','jaki','jakichs','jakichś','jakie','jakis','jakiś','jakiz','jakiż','jakkolwiek','jako','jakos','jakoś','ją','je','jeden','jedna','jednak','jednakze','jednakże','jedno','jego','jej','jemu','jesli','jest','jestem','jeszcze','jeśli','jezeli','jeżeli','juz','już','kazdy','każdy','kiedy','kilka','kims','kimś','kto','ktokolwiek','ktora','ktore','ktorego','ktorej','ktory','ktorych','ktorym','ktorzy','ktos','ktoś','która','które','którego','której','który','których','którym','którzy','ku','lat','lecz','lub','ma','mają','mało','mam','mi','miedzy','między','mimo','mna','mną','mnie','moga','mogą','moi','moim','moj','moja','moje','moze','mozliwe','mozna','może','możliwe','można','mój','mu','musi','my','na','nad','nam','nami','nas','nasi','nasz','nasza','nasze','naszego','naszych','natomiast','natychmiast','nawet','nia','nią','nic','nich','nie','niech','niego','niej','niemu','nigdy','nim','nimi','niz','niż','no','o','obok','od','około','on','ona','one','oni','ono','oraz','oto','owszem','pan','pana','pani','po','pod','podczas','pomimo','ponad','poniewaz','ponieważ','powinien','powinna','powinni','powinno','poza','prawie','przeciez','przecież','przed','przede','przedtem','przez','przy','roku','rowniez','również','sam','sama','są','sie','się','skad','skąd','soba','sobą','sobie','sposob','sposób','swoje','ta','tak','taka','taki','takie','takze','także','tam','te','tego','tej','ten','teraz','też','to','toba','tobą','tobie','totez','toteż','totobą','trzeba','tu','tutaj','twoi','twoim','twoj','twoja','twoje','twój','twym','ty','tych','tylko','tym','u','w','wam','wami','was','wasz','wasza','wasze','we','według','wiele','wielu','więc','więcej','wlasnie','właśnie','wszyscy','wszystkich','wszystkie','wszystkim','wszystko','wtedy','wy','z','za','zaden','zadna','zadne','zadnych','zapewne','zawsze','ze','zeby','zeznowu','zł','znow','znowu','znów','zostal','został','żaden','żadna','żadne','żadnych','że','żeby']
#set of words
location_indicator_tag = ["lokalny","rejon","metr","Street","wokół","koło","łódź","róg","maja","maj","lic","sp","al","wewnątrz","nieopodal","pobliże","koło","otoczenie","okolica","etap","koniec","początek","filia","wschodni","zachodni","południe","południowy","południowozachodni","południowowschodni","północ","północny","północnozachodni","północnowschodni","kierunek","nr","SP","strona","miasto","stacja","osiedlowy","aleja","osiedlać","droga","ul","strefa","ścieżka","odcinek","imienia","estakada","ulica","dzielnica","skrzyżowanie","podziemie","wzdłuż","jan","paweł","ii","szkoła","speaker","atrakcyjność ","cotygodniowy", "weekendowy","obiekt","posesja","projekt","dom","domek","dzień","sala","Mont","chmurka","rolka","punkt","darmowy","obręb","blok","punkt","letni","sensoryczny","miejski","ogólnodostępny","otwarty","plenerowy","publiczny","zewnętrzny","wygodny","sąsiedzki","lokalny","przyjazny","podstawowy","teren", "terenowy", "uliczny", "przestrzeń","cykliczny","chcieć", "istnieć","kontynuacja","edycja","pierwszy","drugi","trzeci","wniosek","cel", "kompleks", "koszt","raz","mini", "nowa", "nowość", "nowy","cykl","wysokość", "zakres","rama", "rzecz","typ","częsty","jaka","rok", "bit","element","miesić","ciąg","czas","godzina","duży","duża","mała","mały","lic"]
#NER with meaning that can be referring to a localisation
location_indicator_ner = ['imię', 'nazwa_geograficzna', 'nazwisko', 'nazwa_instytucji']


class morph_obj:
    def __init__(self, token):
        try:
            self.tok_lem =  re.findall('(\w+)', morf.analyse(token)[0][2][1])[0]
            self.tok_morf = morf.analyse(token)[0][2][2]
            self.tok_ner = morf.analyse(token)[0][2][3]
            self.my_ner = None
        except:
            pass
        
def skip_token(l):
    if l.my_ner == "LOC":
        return False
    if bool(re.search('\_', l.tok_lem)):
        return False
    if bool(re.search('^[A-Za-z]{2}$', l.tok_lem)):
        return False
    
    return True

def process(raw_sentence):
    #tokenize
    sentence = [token.text for token in nlp(raw_sentence)]
    #delete tokens other than alphanumeric and dot
    sentence = [tok for tok in sentence if bool(re.search('[\w\.]', tok))] #delete tokens other than alphanumeric and dot
    #remove roman numerals
    sentence = [tok for tok in sentence if not bool(re.search('^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', tok))] #delete tokens other than alphanumeric and dot
    lem_sent = []
    is_location = False
    is_start_of_sent = True
    for token in sentence:
        try:
                    
            token_obj = morph_obj(token.lower())
            if is_start_of_sent:
                is_start_of_sent = False
            else: #if a word starting with capital letter in the middle of a sentence- most likely is a location
                if bool(re.search('[A-Z]|[ŻŹŁŚĆĄĘÓ]', token[0])) and not is_start_of_sent and not bool(re.search('[A-Z]|[ŻŹŁŚĆĄĘÓ]', token[1])) :
                    token_obj.my_ner = "LOC"
            if bool(re.search('\.+', token)): #end of a sentence
                is_start_of_sent = True
                continue
            # mark token as a location basing on known NER and known words related to location
            if token_obj.tok_lem in location_indicator_tag or  bool(set(token_obj.tok_ner) & set(location_indicator_ner)):
                token_obj.my_ner = "LOC"
            # mark as a location basing on morphems
            if bool(re.search('prep|ign|ncol|pred|conj|comp', token_obj.tok_morf)):
                token_obj.my_ner = "LOC"
            # sportoworekreacyjny/ rekreacyjnosportowy -> sport
            if bool(re.search('sport', token_obj.tok_lem)):
                token_obj.tok_lem = "sport"
            # pieszorowerowy/ rowerowopieszy -> rower
            if bool(re.search('rower', token_obj.tok_lem)):
                token_obj.tok_lem = "rower"
            
            lem_sent.append(token_obj)

        except:
            if len(token)>1:
                print("ups, something slipped...  " + token)
            pass
    try:

        # clean stopwords and lemmatization mistakes     
        sentence = [l.tok_lem for l in lem_sent if skip_token(l)] 
        sentence = [tok for tok in sentence if tok not in stopwords]
        sentence = " ".join(sentence)#store as a string

    except:
        print("sentence lost: "+raw_sentence)
    

    return sentence




def normalization(embeddings):
    norms = np.linalg.norm(embeddings, 2, axis=1, keepdims=True)
    return embeddings/norms

def val_ct_dict(obj):
    unique, counts = np.unique(obj, return_counts=True)
    val_ct = dict(zip(unique, counts))
    val_ct = dict(reversed(sorted(val_ct.items(), key=lambda item: item[1])))
    return val_ct

def embedd(sentence):
    if isinstance(sentence, str):
        embedding = tf.constant([sentence])
    else:
        embedding = tf.constant(sentence)
    embedding = encoderLabse(preprocessor(embedding))["default"]
    embedding = normalization(embedding)
    return embedding

def embedd_list(list_obj):
    embedded = np.zeros((len(list_obj), 768))
    with tqdm(total=len(list_obj)) as pbar:
        for i in range(len(list_obj)):
            embedded[i] = embedd(list_obj[i])
            pbar.update(1)
    return embedded

def val_ct_dict(obj):
    unique, counts = np.unique(obj, return_counts=True)
    val_ct = dict(zip(unique, counts))
    val_ct = dict(reversed(sorted(val_ct.items(), key=lambda item: item[1])))
    return val_ct