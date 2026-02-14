import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
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
  

moby_dick_tokenized = sent_tokenize(clean_text)
print(moby_dick_tokenized[:2])
words = word_tokenize(clean_text)
tagged = pos_tag(words)

stop_words = set(stopwords.words("english"))
stopwords_removed = [word for word in words if word.lower() not in stop_words]

print(tagged)



