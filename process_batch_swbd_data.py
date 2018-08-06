from swda import CorpusReader
from utilities import *
import nltk
nltk.download('averaged_perceptron_tagger')

batch_name = 'dev'  # train, test, val or dev
resource_dir = 'data/'
file_path = resource_dir + batch_name + "_text.txt"
corpus = CorpusReader('switchboard_data/')

# Excluded dialogue act tags
excluded_tags = ['x', '+']

# Process switchboard csv's to text
process_batch_to_txt_file(corpus, resource_dir, batch_name, excluded_tags=excluded_tags)

print("Processing file: ", file_path)
text_data = read_file(file_path)

# Split into labels and sentences
sentences = []
labels = []
for line in text_data:
    sentences.append(line.split("|")[0])
    labels.append(line.split("|")[1])

# Generate tokenised utterances
utterances = []
for sentence in sentences:
    utterances.append(nltk.word_tokenize(sentence))

# Save data to file
data = dict(
    utterances=utterances,
    labels=labels)

save_data(resource_dir + batch_name + "_data.pkl", data)
