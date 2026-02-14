import re
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.chunk import RegexpParser

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

file = open('./moby-dick.txt')
text_from_file = file.read()
file.close()

clean_text = re.sub(r'\d+', "", text_from_file)
clean_text = re.sub(r'\s+', " ", clean_text)
clean_text = re.sub(r'[^a-zA-Z0-9. ]', "", clean_text)
clean_text = re.sub(r'\.{2}', '.', clean_text)

moby_dick_tokenized = sent_tokenize(clean_text)
# print(moby_dick_tokenized[:2])
words = word_tokenize(clean_text)
tagged = pos_tag(words)

lemma = WordNetLemmatizer.lemmatize(words, pos='v')

stop_words = set(stopwords.words("english"))
stopwords_removed = [word for word in words if word.lower() not in stop_words]

chunk_grammar = r"""
  NP: {<DT>?<JJ.*>*<NN.*>+}         # Noun Phrases
  PP: {<IN><NP>}                    # Prepositional Phrases
  VP: {<VB.*><NP|PP|CLAUSE>*}       # Verb Phrases
  ADJP: {<RB.*>*<JJ.*>}             # Adjective Phrases
  ADVP: {<RB.*>+}                   # Adverb Phrases
"""
parser = RegexpParser(chunk_grammar)
chunked_tree = parser.parse(tagged)

# Lemmatization helper (convert POS format)
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN


# 6. Lemmatize words
def lemmatize_words(tagged_words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = []

    for sentence_tags in tagged_words:
        lemmatized_sentence = []
        for word, tag in sentence_tags:
            wordnet_pos = get_wordnet_pos(tag)
            lemmatized_sentence.append(
                lemmatizer.lemmatize(word, wordnet_pos)
            )
        lemmatized_sentences.append(lemmatized_sentence)

    return lemmatized_sentences



